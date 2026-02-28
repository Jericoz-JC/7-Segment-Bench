import os
import platform

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, 'data', 'uploads')
    THUMBNAIL_FOLDER = os.path.join(basedir, 'data', 'thumbnails')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload
    THUMBNAIL_SIZE = (256, 256)
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

    IS_PI = platform.machine().startswith('aarch64') or platform.machine().startswith('arm')
    USE_NCNN = IS_PI


class DevConfig(Config):
    DEBUG = True


class PiConfig(Config):
    DEBUG = False
    USE_NCNN = True


config_map = {
    'dev': DevConfig,
    'pi': PiConfig,
    'default': DevConfig,
}
