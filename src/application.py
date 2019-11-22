from src.util.hanspell.spell_checker import fix
from src.util.tokenizer import tokenize
from src.intent.classifier import get_intent

def run():
    while True:
        print('User: ', sep='', end='')
        speech = preprocess(input())
        print('Preprocessed : ' + speech , sep='')
        intent = get_intent(speech)
        print('Intent : ' + intent , sep='')



def preprocess(speech) ->str:
    speech = fix(speech)
    speech = tokenize(speech)
    speech = fix(speech)
    return speech