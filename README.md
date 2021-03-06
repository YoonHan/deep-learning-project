# 2019 Fall Deep Learning Project

2019F Deep Learning Class Project

**Goal :** Angry Birds 게임의 Autoplay AI agent가 게임의 각 level 에서 최소 별 2개 이상의 점수를 받을 수 있도록 학습시킨다.

**\[ Developmnet Environment \]**

`HOST: Linux Ubuntu 16.04 LTS` 
`Java version: OpenJDK 1.8.0_222` 

---

### Server-Client architecture

Chrome extension <-> Server <-> Client  
구조로 되어있다.   Game Play를 진행하는 AI agent 는 Client 에 속해 있으며, server 에 특정 명령(Shoot) 을 전송하면 server가 proxy module 을 통해 chrome extension 인 angry birds extension 과 통신하여 client 로부터 전달받은 명령을 수행하는 과정을 반복한다. client 로부터 받은 명령을 수행하고 나면 그 결과를 받아서 다시 client로 전달해준다. client 는 server로 명령을 전달할 때 message 를 통해 전달하는데, 이러한 message의 종류는 크게 4가지로 나눌 수 있다. - configuration  - query - in-game  - level selection

### Vision module

2 개의 image segmentation component 로 구성되어 있다. 하나는 게임 스크린샷을 보고 중요한 object 들의 Minimum Bounding Rectangles(MBR)을 찾아내는 module 이다. 게임 내의 중요한 object들의 종류에는 sling, red bird, yellow bird, blue bird, black bird, white bird, pig, ice, wood, stone, TNT, TrajectoryPoint 가 있다. 또 하나의 component 는 MBR이 아닌 objects 의 real shape 를 구해준다. 스크린샷을 보고 분석할 때 두 가지 component를 모두 사용할 수 있다.

### Trajectory module

trajectory module 에서는 새를 쐈을 때 그 새가 날아갈 궤적을 계산해주는 모듈이다. Newton의 공식을 통해서 parabolic path 를 구하는데, 각 궤적들에 속하는 포인트들의 list를 반환해준다.

## Create your own Intelligent Agent

vision module은 game play screenshot 에서 object 들에 대한 MBR을 ABObject 로 반환한다. 이 object는 x, y좌표 및 width, height 값을 포함한다. 이 값들 말고도 해당 object가 어떤 type의 object인지도 알려준다.
\*\* MBR segmentation 은 hill object를 감지하지 못하고, Real Shape segmentation 은 TNT를 감지하지 못한다. 이 점을 참고여야 한다.

## Access the Game state

5 가지의 game state 가 있다.

1. WON
2. LOST
3. PLAYING
4. LEVEL SELECTINGS
5. LOADING

## Compile

ANT 사용

build.xml 파일이 있는 디렉토리에서

`ant compile`
`ant jar`

명령을 순서대로 실행한다.

## 기본 소스코드 실행 로직 분석

1. `MainEntry.java` 에서 프로그램 실행 시 인자로 넘겨준 옵션에 따라 `NaiveAgent` 실행 방식을 다르게 해준다.

2. `-na` 옵션만주고 실행시켰을 때, `MainEntry.java` 에서 `na.run()` 코드가 실행된다.

3. `aRobot.loadLevel(currentLevel)` 코드로 레벨(맵)을 불러온 후에 `while` 무한루프로 게임을 실행시킨다.

4. `GameState state = solve()` 에서 현재 game state 를 구한다. `solve()` 에는 현재 게임 화면을 분석해서 object detection 을 한 후에 detect 된 pig 를 랜덤하게 골라 targeting 한다.

5. 해당 pig 를 맞출 수 있는 trajectory 를 찾아서 새총을 발사한다.

## 기본 소스코드에서 변경된 로직 설명

---

## Reinforcement Learning

Agent 를 학습시키는 방식에는

- Policy learning
- Q-Learning

두 가지가 있다.

**Policy learning** 은 "적을 만났는데, 너보다 세면 튀어라" 같은 policy 들을 학습시키는 방법이다.

**Q-Learning** 은 policy learning 과는 다르게 policy를 제공하지 않고 두개의 input인 state, action을 받는다. 이 input pair를 받아서, 각 pair 마다 어떤 값을 계산해내게 되는데 이 값은 각 state에서 agent가 어떤 action을 취할지에 대한 기대값이다.

본 프로젝트에서는 **Q-Learning** 을 사용한다.
-> 원래 Q-Learning 을 사용해보려고 했으나, 본 프로젝트의 action space 가 discrete 이 아니라 continuous space 이므로
deep deterministic policy gradient(DDPG)를 사용하기로 했다.

출처: https://algorithmia.com/blog/introduction-to-reinforcement-learning

---

## 참고

### Reinforcement Learning

- [강화학습 기초](https://daeson.tistory.com/311?category=710652)

- [강화학습 알아보기](https://greentec.github.io/reinforcement-learning-fourth/)

- [A3C 쉽고 깊게 이해하기](https://www.slideshare.net/WoongwonLee/rlcode-a3c)

- [Simple implementation in RL](https://www.youtube.com/watch?v=yMk_XtIEzH8)

- [DDPG](https://reinforcement-learning-kr.github.io/2018/06/26/3_ddpg/)

## 프로젝트 세팅 및 빌드

- [ANT](https://ant.apache.org/)
