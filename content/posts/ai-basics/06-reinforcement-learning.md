---
title: "강화학습의 기초: 환경과의 상호작용"
date: 2024-05-19
description: "강화학습의 기본 개념과 주요 알고리즘에 대해 알아봅니다."
tags: ["AI", "강화학습", "RL"]
---

# 강화학습의 기초: 환경과의 상호작용

## 강화학습이란?

강화학습(Reinforcement Learning)은 에이전트가 환경과 상호작용하며 보상을 최대화하는 행동을 학습하는 기술입니다. 게임 AI, 로봇 제어, 자율주행 등 다양한 분야에서 활용됩니다.

## 강화학습의 기본 요소

### 1. 환경(Environment)
- 에이전트가 상호작용하는 세계
- 상태(State)와 보상(Reward)을 제공

### 2. 에이전트(Agent)
- 환경과 상호작용하는 주체
- 행동(Action)을 선택하고 학습

### 3. 보상(Reward)
- 행동의 결과에 대한 피드백
- 장기적인 보상을 최대화하는 것이 목표

## 간단한 강화학습 예제

### 1. Q-Learning 구현
```python
import numpy as np

class QLearning:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9):
        self.q_table = np.zeros((states, actions))
        self.lr = learning_rate
        self.gamma = discount_factor
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value
    
    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
```

### 2. 간단한 환경 구현
```python
class SimpleEnvironment:
    def __init__(self, size=5):
        self.size = size
        self.state = 0
        self.goal = size - 1
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        if action == 0:  # 왼쪽
            self.state = max(0, self.state - 1)
        else:  # 오른쪽
            self.state = min(self.size - 1, self.state + 1)
        
        done = self.state == self.goal
        reward = 1 if done else 0
        
        return self.state, reward, done
```

## DQN (Deep Q-Network)

### 1. 신경망 모델 정의
```python
import tensorflow as tf

def create_dqn_model(input_shape, num_actions):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
```

### 2. DQN 에이전트 구현
```python
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_dqn_model((state_size,), action_size)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
```

## 강화학습의 응용 분야

1. **게임 AI**
   - 알파고
   - Dota 2 AI
   - 스타크래프트 AI

2. **로봇 제어**
   - 로봇 팔 제어
   - 보행 로봇
   - 드론 제어

3. **자율주행**
   - 경로 계획
   - 충돌 회피
   - 교통 신호 준수

## 강화학습의 주요 알고리즘

1. **Value-based Methods**
   - Q-Learning
   - DQN
   - Double DQN

2. **Policy-based Methods**
   - Policy Gradient
   - Actor-Critic
   - PPO

3. **Model-based Methods**
   - Dyna-Q
   - Model-based Policy Optimization

## 강화학습의 한계와 과제

1. **샘플 효율성**
   - 많은 시도가 필요
   - 학습 시간이 김
   - 실험 비용이 큼

2. **안정성**
   - 학습이 불안정할 수 있음
   - 하이퍼파라미터에 민감
   - 수렴이 보장되지 않음

## 최신 트렌드

1. **Multi-Agent RL**
   - 협력 학습
   - 경쟁 학습
   - 분산 학습

2. **Hierarchical RL**
   - 계층적 정책
   - 장기 계획
   - 추상화된 행동

## 결론

강화학습은 AI 분야에서 가장 활발하게 연구되고 있는 기술 중 하나입니다. 환경과의 상호작용을 통해 학습하는 방식은 인간의 학습 과정과 유사하며, 다양한 분야에서 혁신적인 변화를 가져오고 있습니다.

다음 포스트에서는 생성형 AI의 기초에 대해 알아보도록 하겠습니다. 