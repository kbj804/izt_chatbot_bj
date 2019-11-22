# Author : Hyunwoong
# When : 5/6/2019
# Homepage : github.com/gusdnd852
import os

import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow.contrib.eager as tfe
from gensim.models import FastText
from konlpy.tag import Okt

from src.intent.configs import IntentConfigs

configs = IntentConfigs()
# 파라미터 세팅
data = configs.data
encode_length = configs.encode_length
label_size = configs.label_size
filter_sizes = configs.filter_sizes
num_filters = configs.num_filters
intent_mapping = configs.intent_mapping
learning_step = configs.learning_step
learning_rate = configs.learning_rate
vector_size = configs.vector_size

## tf버전 문제 
#tf = tf.compat.v1
tf.disable_eager_execution()


def inference_embed(data):
    mecab = Okt()
    # FastText: Google에서 개발한 Word2Vec을 기본으로 부분단어들을 임베딩
    model = FastText.load(configs.fasttext_path +'model')
    encode_raw = mecab.morphs(data)
    encode_raw = list(map(lambda x: encode_raw[x] if x < len(encode_raw) else '#', range(encode_length)))
    input = np.array(
        list(map(lambda x: model[x] if x in model.wv.index2word else np.zeros(vector_size, dtype=float), encode_raw)))
    return input


def create_graph(train=True):
    #플레이스홀더는 데이터를 입력받는 비어있는 변수
    x = tf.placeholder("float", shape=[None, encode_length * vector_size], name='x')
    y_target = tf.placeholder("float", shape=[None, label_size], name='y_target')
    # tf.reshape: -1은 자동/ 차원의 갯수와 크기를 설정 
    # 마지막 차원은 1개, 이전차원 vector_size개 , 그 이전차원 encode_length개 인 4차원 행렬
    x_image = tf.reshape(x, [-1, encode_length, vector_size, 1], name="x_image")


    # Const는 n차원 데이터 타입이라 각하면 편함 tf에서 사용하는 변수 선언 정도 개념
    # 변수에 접근하기 위해서는 Session을 사용해야 함.
    #l2_loss = tf.constant(0.0)
    """
    Convoultion and Max-Pooling Layers를 만드는 부분. 
    각 크기가 다른 필터를 사용하여 반복적으로 합성곱 텐서를 생성하고 이를 하나의 큰 피쳐백터로 병합
    """
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        # name_scope : 그래프의 노드가 많을 경우 쉽게 볼 수 없기 때문에 단순화 필요
        # 이름을 가진 각 노드들의 범주를 지정해 줄 수 있고 계층 구조의 맨 위만 표시됨.
        # name_scope가 넓을수록 시각화가 잘 됨
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, vector_size, 1, num_filters]
            # truncated_normal : 절단정규분포의 난수값 생성 -> 정규분포에서 너무 벗어난 이상치 제거용도
            # stddev: 절단정규분포의 표준편차
            W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W") # W: 필터 행렬
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            
            #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
            # for CNN
            conv = tf.nn.conv2d(
                x_image,
                W_conv1,
                strides=[1, 1, 1, 1],
                padding="VALID", # 엣지 패딩 없이 문장을 슬라이드하여 strides 크기로 좁은(narrow) 합성곱을 수행
                name="conv")

            """
            렐루(ReLU, Rectified Linear Unit) 활성화 함수는 최근 딥 뉴럴 네트워크의 히든 레이어에서 사용되는 기본 활성화 함수가 되었습니다. 
            이 함수는 간단한데 max(0, x) 로서 음수의 경우 0 을 리턴하고 그 외에는 x 를 리턴합니다. 
            이 예제에서는 콘볼루션 히든 레이어의 활성화 함수로 렐루 함수를 사용하겠습니다.
            """
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv1), name="relu") #h: 합성곱 출력에 비선형성(ReLU) 적용 결과
            # 풀링은 콘볼루션 처럼 가로, 세로 일정 간격으로 특성 맵을 스캔
            # 하지만 콘볼루션처럼 필터를 곱하는 것이 아니고 특성 맵의 값을 평균 낸다거나 가장 큰 값을 뽑아서 사용
            # max_pool은 최대값을 선택함 (서브샘플링 개념. 치를 곱하거나 바이어스를 더하는거 안함)
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, encode_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name="pool")
            #각 필터 사이즈의 맥스 풀링 출력은 [batch_size, 1, 1, num_filters]가 되며 이것이 최종 피쳐에 대응하는 마지막 피쳐 벡터
            pooled_outputs.append(pooled)

    # 모든 풀링 벡터는 [batch_size, num_filters_total] 모양을 갖는 하나의 긴 피쳐 벡터로 결합
    num_filters_total = num_filters * len(filter_sizes)
    # 3번째 axis 제거 후 합침
    h_pool = tf.concat(pooled_outputs, 3)

    # h_pool_flat : 아래에 Dropout을 적용할 텐서
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) #텐서를 평평하게.. -1
    keep_prob = 1.0
    if train:
        # 주어진 유닛을 유지할 확률 kepp_prob -> drop지 않을 확률
        keep_prob = tf.placeholder("float", name="keep_prob")
        # Dropout은 over-fitting을 줄이기 위한 regularization 기술
        # 네트워크에서 일시적으로 유닛(인공 뉴런, artificial neurons)을 배제하고, 그 배제된 유닛의 연결을 모두 끊는다.
        h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[num_filters_total, label_size]), name="W_fc1")
    #W_fc1 = tf.get_variable(
     #   "W_fc1",
      #  shape=[num_filters_total, label_size],
       # initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[label_size]), name="b")

    # 보통 오버피팅 억제용으로 사용하긴함 근데여기선 안쓰는데..l2_loss
    #l2_loss += tf.nn.l2_loss(W_fc1)
    #l2_loss += tf.nn.l2_loss(b_fc1)

    # 맥스 풀링(드롭아웃이 적용된 상태에서)으로 피쳐 벡터를 사용하여 행렬 곱셈을 수행하고 가장 높은 점수로 분류를 선택하는 예측을 수행
    y = tf.nn.xw_plus_b(h_pool_flat, W_fc1, b_fc1, name="scores")
    predictions = tf.argmax(y, 1, name="predictions")

    # 점수를 이용해 손실 함수를 정의한다. 손실은 망에서 발생하는 오류를 나타내는 척도이며 이를 최소화 하는게 우리의 목표다. 
    # 분류 문제에 대한 표준 손실 함수는 cross-entropy loss를 사용한다.
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_target)
    cross_entropy = tf.reduce_mean(losses)

    # TextCNN 모델을 인스턴스화한 다음 망의 손실 함수를 최적화하는 방법을 정의한다. 
    # 텐서플로우에는 여러가지 옵티마이저가 내장되어 있는데 여기서는 아담 옵티마이저를 사용
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # tf.equal(A,B) : A와 B를 비교하여 Boolean 값 반환
    correct_predictions = tf.equal(predictions, tf.argmax(y_target, 1))

    # reduce_mean : 배열의 평균, cast : 자료형 변환
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    return accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1


def predict(test_data):
    try:
        # Default Graph로 초기화.
        tf.reset_default_graph()
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        sess = tf.Session()
        _, x, _, _, _, y, _, _ = create_graph(train=False)

        # 생성한 변수를 입력하여 초기화
        sess.run(tf.global_variables_initializer())

        # 모델과 파라미터 저장.
        saver = tf.train.Saver()
        dir = os.listdir(configs.model_path)
        num_ckpt = 0
        for i in dir:
            try:
                new_one = int(i.split('-')[1].split('.')[0])
                if num_ckpt < new_one:
                    num_ckpt = new_one
            except:
                pass
        
        # 학습된 값들을 불러옴 restore
        saver.restore(sess, configs.model_path + 'check_point-' + str(num_ckpt) + '.ckpt')
        y = sess.run([y], feed_dict={x: np.array([test_data])})
        score = y[0][0][np.argmax(y)]
        print("MAX Score: ",score)
        if score > configs.fallback_score:
            print(format(np.argmax(y)))
            return format(np.argmax(y))
        else:
            print(format(np.argmax(y)))
            return None
    except Exception as e:
        raise Exception("error on training: {0}".format(e))
    finally:
        sess.close()


def get_intent(text):
    prediction = predict(np.array(inference_embed(text)).flatten())
    if prediction is None:
        return "폴백"
    else:
        for mapping, num in intent_mapping.items():
            if int(prediction) == num:
                return mapping
