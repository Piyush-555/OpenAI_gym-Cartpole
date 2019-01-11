import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from collections import Counter
from statistics import mean

def generate_data(n_episodes):
    accepted_scores=[]
    train_data=[]

    for i_episode in range(n_episodes):
        score=0
        prev_observation = env.reset()
        temp=[]
        for timesteps in range(500):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            temp.append([prev_observation,action])
            prev_observation=observation
            score+=reward
            if done:
                break
        if score>=50:
            accepted_scores.append(score)
            for data in temp:
                if data[1]==1:
                    y=[0,1]
                else:
                    y=[1,0]
                train_data.append([data[0],y])

    train_data=np.array(train_data)
    np.save('train.npy',train_data)
    print("Average rewards:",mean(accepted_scores))
    return train_data

def neural_network(n_input):
    net = tflearn.input_data(shape=[None, n_input, 1])
    net = tflearn.fully_connected(net, 32, activation='relu')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 16, activation='relu')
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model

def train_model(train_data, n_input):
    X=np.array([i[0] for i in train_data]).reshape(-1, 4, 1)
    Y=[i[1] for i in train_data]
    model.fit(X, Y, n_epoch=3, show_metric=True, run_id='first_ai_game')    
    return model

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    data=generate_data(10000)
    print('Generating Data Completed...')
    model=neural_network(4)
    model=train_model(data, 4)
    print('Model Train Completed...')
    #model.load('model_type2_197.model')
    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        while True:
            env.render()
            if len(prev_obs)==0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,4,1))[0])
            choices.append(action)         
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done:
                break
        scores.append(score)
        print('Episode {} Score: {}'.format(each_game+1,score))

    print('Average Score:',sum(scores)/len(scores))
    


