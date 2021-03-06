#
# Cf. https://github.com/dconathan/madpy-deploy-ml/blob/master/src/app.py
#

import const
import falcon
import logging
from model import translate_fren
import os
import random
import sys
from wsgiref import simple_server

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


api = falcon.API()


def load_lines(fn):
    with open(fn, 'r') as f:
        lines = [line for line in f]
    return lines


class Health:
    """
    Returns the message 'OK' to prove the server is working.
    """
    def on_get(self, req: falcon.Request, resp: falcon.Response):
        resp.body = "OK"
        resp.content_type = falcon.MEDIA_TEXT


class TranslateFREN:
    """
    Translates a sentence from French to English.
    """
    def on_get(self, req: falcon.Request, resp: falcon.Response):
        resp.body = self.on_post.__doc__
        resp.content_type = falcon.MEDIA_TEXT

    def on_post(self, req: falcon.Request, resp: falcon.Response):
        """
        POST /api/translate/fren

        Translate french sentence to english (70 word limit).

        Input:  French text (UTF-8 string)
        Output: English text (UTF-8 string)

        Example input:
        {
            "fr": "Bonjour le monde!"
        }
        Example output:
        {
            "en": "Hello world!"
        }
        """

        fr_text = req.media.get("fr")

        if fr_text is None:
            raise falcon.HTTPMissingParam("fr")

        if not isinstance(fr_text, str):
            raise falcon.HTTPInvalidParam(
                f"expected a string, got a {type(fr_text).__name__}", "fr"
            )

        resp.media = {"en": translate_fren(fr_text)}


class ExampleFR:
    """
    Generate an example French sentence for translation.
    """
    examples = load_lines(const.EXAMPLES_FILE)

    def on_get(self, req: falcon.Request, resp: falcon.Response):
        example = random.sample(self.examples, 1)[0]
        resp.body = example
        resp.content_type = falcon.MEDIA_TEXT


class Root:
    """
    Displays the translation webpage at the root URL.
    """
    def on_get(self, req: falcon.Request, resp: falcon.Response):
        raise falcon.HTTPTemporaryRedirect('/index.html')


api.add_route('/api/translate/fren', TranslateFREN())
api.add_route('/api/example/fr', ExampleFR())
api.add_route('/health', Health())
api.add_route('/', Root())
api.add_static_route('/', os.path.join(const.APP_DIR, 'src', 'www'))

if 'pytest' not in sys.modules:
    logger.info('warming up model')
    translate_fren('Réchauffez-vous!')
    logger.info('model is ready')

if __name__ == '__main__':
    httpd = simple_server.make_server('0.0.0.0', 8080, api)
    httpd.serve_forever()
