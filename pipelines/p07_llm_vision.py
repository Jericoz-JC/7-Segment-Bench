"""Pipeline 7: LLM Vision API (Claude or GPT-4o)."""

import base64
import re
import cv2
import numpy as np
from pipelines import register
from pipelines.base import BasePipeline, PipelineResult, ROI
from pipelines.preprocessing import crop_roi


def _encode_image(image: np.ndarray) -> str:
    """Encode image as base64 JPEG."""
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode('utf-8')


def _extract_digits(text: str) -> str:
    """Extract digit sequence from LLM response."""
    # Look for explicit digit patterns
    patterns = [
        r'(?:digits?|number|reading|display|value)[:\s]*["\']?(\d+)["\']?',
        r'(\d{2,})',  # 2+ digits in a row
        r'(\d)',  # individual digits
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return ''.join(matches)
    return ''


def _extract_openai_text(content) -> str:
    """Normalize OpenAI-style message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get('text')
                if txt:
                    parts.append(str(txt))
        return ''.join(parts)
    return str(content or '')


@register
class LLMVisionPipeline(BasePipeline):
    name = 'LLM Vision API'
    slug = 'p07_llm_vision'
    description = 'Claude or GPT-4o vision; zero-shot digit reading'
    is_online = True
    is_trainable = False
    _supported_providers = ('anthropic', 'openai', 'openrouter', 'ollama')

    def __init__(self, config=None):
        super().__init__(config)
        self._client = None
        self._provider = None

    def _default_model(self, provider: str) -> str:
        defaults = {
            'anthropic': 'claude-sonnet-4-20250514',
            'openai': 'gpt-4o',
            'openrouter': 'openrouter/auto',
            'ollama': 'llava',
        }
        return defaults.get(provider, 'gpt-4o')

    def load(self):
        super().load()
        # Try to get API key from config or DB
        provider = str(self.config.get('provider', 'anthropic')).strip().lower()
        if provider not in self._supported_providers:
            raise ValueError(
                f'Unknown provider: {provider}. '
                f'Choose one of {list(self._supported_providers)}'
            )

        api_key = str(self.config.get('api_key', '')).strip()

        if not api_key:
            # Try loading from database
            try:
                from flask import current_app
                with current_app.app_context():
                    from models.benchmark import ApiKey
                    key_record = ApiKey.query.filter_by(
                        provider=provider, is_active=True
                    ).first()
                    if key_record:
                            api_key = key_record.key_value
            except Exception:
                pass

        if provider != 'ollama' and not api_key:
            raise ValueError(f'No API key configured for {provider}. '
                             f'Add one via Settings or pass in pipeline config.')

        self._provider = provider
        if provider == 'anthropic':
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif provider == 'openai':
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        elif provider == 'openrouter':
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url='https://openrouter.ai/api/v1',
            )
        elif provider == 'ollama':
            import openai
            ollama_base_url = str(self.config.get('base_url', 'http://localhost:11434/v1')).strip()
            self._client = openai.OpenAI(
                api_key=api_key or 'ollama-local',
                base_url=ollama_base_url,
            )
        else:
            raise ValueError(f'Unknown provider: {provider}')

    def unload(self):
        self._client = None
        super().unload()

    def predict(self, image: np.ndarray, roi: ROI) -> PipelineResult:
        if self._client is None:
            return PipelineResult(error='LLM client not loaded')

        cropped = crop_roi(image, roi)
        b64 = _encode_image(cropped)

        prompt = (
            "This image shows a 7-segment LED/LCD display. "
            "Read the digits shown on the display. "
            "Respond with ONLY the digit characters you see, nothing else. "
            "For example: 0237"
        )

        try:
            if self._provider == 'anthropic':
                response = self._client.messages.create(
                    model=self.config.get('model', self._default_model(self._provider)),
                    max_tokens=50,
                    messages=[{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image',
                                'source': {
                                    'type': 'base64',
                                    'media_type': 'image/jpeg',
                                    'data': b64,
                                }
                            },
                            {'type': 'text', 'text': prompt}
                        ]
                    }]
                )
                text = response.content[0].text
            else:  # OpenAI-compatible providers: openai/openrouter/ollama
                response = self._client.chat.completions.create(
                    model=self.config.get('model', self._default_model(self._provider)),
                    max_tokens=50,
                    messages=[{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/jpeg;base64,{b64}'
                                }
                            },
                            {'type': 'text', 'text': prompt}
                        ]
                    }]
                )
                text = _extract_openai_text(response.choices[0].message.content)

            predicted = _extract_digits(text)
            return PipelineResult(
                predicted=predicted,
                confidence=0.9 if predicted else 0.0,
                debug_images={'input': cropped},
            )

        except Exception as e:
            return PipelineResult(
                error=f'API error: {e}',
                debug_images={'input': cropped},
            )
