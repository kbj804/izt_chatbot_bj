# -*- coding: utf-8 -*-

def main(path):
    # 작업 디렉토리의 path를 추가
    import sys
    sys.path.append(path)
    import src.application as app
    app.run()

def clear_log():
    import logging
    import os
    import tensorflow.compat.v1 as tf

    path = os.getcwd()
    logger = logging.getLogger('chardet')
    logger.setLevel(logging.CRITICAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    return path

if __name__ == '__main__':
    print("AI is awakening now....")
    print("Provided Feature : ")
    path = clear_log()
    #print("Current Directory PATH: ",path)
    main(path)

""" 참고용
import os
print (os.getcwd()) #현재 디렉토리의
print (os.path.realpath(__file__))#파일
print (os.path.dirname(os.path.realpath(__file__)) )#파일이 위치한 디렉토리
"""
