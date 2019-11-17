# deep-learning-project

2019F Deep Learning Class Project

### Server-Client architecture

Chrome extension <-> Server <-> Client 구조로 되어있다.  Game Play를 진행하는 AI agent 는 Client 에 속해 있으며, server 에 특정 명령(Shoot) 을 전송하면 server가 proxy module 을 통해 chrome extension 인 angry birds extension 과 통신하여 client 로부터 전달받은 명령을 수행하는 과정을 반복한다. client 로부터 받은 명령을 수행하고 나면 그 결과를 받아서 다시 client로 전달해준다. client 는 server로 명령을 전달할 때 message 를 통해 전달하는데, 이러한 message의 종류는 크게 4가지로 나눌 수 있다. - configuration  - query - in-game  - level selection

### Vision module

2 개의 image segmentation component 로 구성되어 있다. 하나는 게임 스크린샷을 보고 중요한 object 들의 Minimum Bounding Rectangles(MBR)을 찾아내는 module 이다. 게임 내의 중요한 object들의 종류에는 sling, red bird, yellow bird, blue bird, black bird, white bird, pig, ice, wood, stone, TNT, TrajectoryPoint 가 있다. 또 하나의 component 는 MBR이 아닌 objects 의 real shape 를 구해준다. 스크린샷을 보고 분석할 때 두 가지 component를 모두 사용할 수 있다.

## Trajectory module

trajectory module 에서는 새를 쐈을 때 그 새가 날아갈 궤적을 계산해주는 모듈이다. Newton의 공식을 통해서 parabolic path 를 구하는데, 각 궤적들에 속하는 포인트들의 list를 반환해준다.

## Create your own Intelligent Agent

vision module은 game play screenshot 에서 object 들에 대한 MBR을 ABObject 로 반환한다. 이 object는 x, y좌표 및 width, height 값을 포함한다. 이 값들 말고도 해당 object가 어떤 type의 object인지도 알려준다.
\*\* MBR segmentation 은 hill object를 감지하지 못하고, Real Shape segmentation 은 TNT를 감지하지 못한다. 이 점을 참고여야 한다.

## Access the Game state

4 가지의 game state 가 있다.

1. WON
2. LOST
3. PLAYING
4. LEVEL SELECTINGS
5. LOADING

## Compile

ANT 사용
