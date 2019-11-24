# -*- coding: utf-8 -*-
from jpype import *
import os.path
import numpy as np
import random
from ddpg import DDPG
from ou_noise import OUNoise

# Jpype Setting.
# Jpype: enable to use Java class library into python codebase.
jarpath = os.path.abspath('../../')
startJVM(getDefaultJVMPath(), "-Djava.ext.dirs=%s" % jarpath)
TrajectoryPkg = JPackage('ab').planner
DemoOtherPkg = JPackage('ab').demo.other
VisionPkg = JPackage('ab').vision
AwtPkg = JPackage('java').awt
UtilPkg = JPackage('java').util

# Load Java inner package
Point = AwtPkg.Point
Rectangle = AwtPkg.Rectangle
BufferedImage = AwtPkg.image.BufferedImage
ArrayList = UtilPkg.ArrayList
List = UtilPkg.List

# Get Game moudle implemented by Java
TrajectoryPlanner = TrajectoryPkg.TrajectoryPlanner
ClientActionRobot = DemoOtherPkg.ClientActionRobot
ClientActionRobotJava = DemoOtherPkg.ClientActionRobotJava
ABObject = VisionPkg.ABObject
GameStateExtractor = VisionPkg.GameStateExtractor
Vision = VisionPkg.Vision

# 학습에 사용할 파라미터 정의
episodes = 10000
is_batch_norm = False
TRAIN_STEP = 50

# Agent class


class PlayAgent:

    def __init__(self, ip="127.0.0.1", id=28888):   # ip 는 서버 주소.id 는 agent 식별자
        self.ar = ClientActionRobotJava(ip)
        self.se = GameStateExtractor()  # ㅎ믇
        self.tp = TrajectoryPlanner()   # 궤적 계산 모듈
        self.firstShot = True
        self.solved = []            # clear 한 레벨은 1, 아닌 레벨은 0의 값을 가진다
        self.currentLevel = -1
        self.failedCounter = 0
        self.id = id
        self.width = 840            # 게임 화면 너비
        self.height = 480           # 게임 화면 높이
        # 게임 화면 스크린샷(=state) 의 크기. 즉. state의 개수는
        # 화면의 픽셀 수와 같다.
        # [height, width, RGB 3 channels]
        self.num_states = [self.height, self.width, 3]
        # Action space 정의
        # [거리(0~90px), 각도(0~90degree), tapTime(0~5000ms)]
        self.num_actions = 3
        self.action_space_high = [90, 75, 50]
        self.action_space_low = [0, 0, 0]
        self.noise_mean = [20, -20, 0]
        self.noise_sigma = [10, 30, 20]
        self.ddpg = DDPG(self.num_states, self.num_actions,
                         self.action_space_high, self.action_space_low, is_batch_norm)

    def getNextLevel(self):     # 다음 레벨을 얻어온다

        level = 0
        unsolved = False

        for i in range(len(self.solved)):
            if self.solved[i] == 0:
                unsolved = True
                level = i + 1
                if level <= self.currentLevel and self.currentLevel < len(self.solved):
                    continue
                else:
                    return level

        if unsolved:
            return level

        level = (self.currentLevel + 1) % len(self.solved)
        if level == 0:
            level = len(self.solved)

        return level

    def checkMyScore(self):

        scores = self.ar.checkMyScore()     # 현재 점수 확인
        level = 1
        for s in scores:    # 각 level 별 점수 확인
            print "||\tlevel %d score is : %d\t||" % (level, s)
            if s > 0:
                self.solved[level - 1] = 1
            level += 1

    def getScreenBuffer(self, buffer, width=840, height=480):
        """
            현재 게임플레이 스크린샷을 받아온다.
            RGB 별로 따로 저장한다.
        """
        print "## Get ScreenBuffer"
        # returnBuffer's size = (480, 840, 3)
        returnBuffer = np.zeros((height, width, 3))
        for i in range(height):
            for j in range(width):
                RGB = buffer.getRGB(j, i)
                returnBuffer[i, j, 0] = RGB & 0x0000ff
                returnBuffer[i, j, 1] = RGB & 0x00ff00
                returnBuffer[i, j, 2] = RGB & 0xff0000

        print "## Return ScreenBuffer"
        return returnBuffer

    def shoot(self, action):
        """
            새를 쏘고,
            쏜 후의 상태를 반환한다.
        """
        # 새총 detection
        screenshot = self.ar.doScreenShot()
        vision = Vision(screenshot)
        sling = vision.findSlingshotMBR()

        # 현재 게임 state
        pigs = vision.findPigsMBR()
        state = self.ar.checkState()

        # 새총이 감지되면 플레이하고, 아니라면 스킵
        if sling != None:

            # 맵에 돼지가 존재하면 임의로 한 마리를 타겟으로 잡고 쏜다.
            if len(pigs) != 0:

                refPoint = self.tp.getReferencePoint(sling)
                print "## Ref Sling Point : ", refPoint

                # DDPG 로부터 취할 action을 받아온다
                releaseDistance = action[0]
                releaseAngle = action[1]
                tapTime = action[2]
                print "## Release Distance : ", releaseDistance
                print "## Release Angle : ", releaseAngle

                self.ar.fullyZoomOut()
                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                _sling = vision.findSlingshotMBR()  # zoom out 했을 때 감지된 새총.

                if _sling != None:
                    # zoom out 하지 않았을 때의 새총 위치와 zoom out 한 후의 새총 위치의 차이를 구하여
                    # 너무 차이가 난다면, 쏘지 않고 다시 screenshot 을 찍어 분석하도록 함
                    scale_diff = (sling.width - _sling.width) ** 2 + \
                        (sling.height - _sling.height) ** 2

                    if scale_diff < 25:
                        self.ar.shoot(int(refPoint.x), int(refPoint.y), int(
                            releaseDistance), int(releaseAngle), 0, int(tapTime), True)
                        print "## Shooting is Done"
                        state = self.ar.checkState()

                        if state == state.PLAYING:
                            self.firstShot = False

                    else:
                        print "## Scale is changed. So sling can not execute the shot and will re-segment the image"
                else:
                    print "## No sling was detected. So agent can not execute the shot and will re-segment the image"

        return state

    def ddpg_run(self):
        """
            DDPG algorithm 을 raw pixel data(screenshot)에 대해서 돌린다
        """

        info = self.ar.configure(ClientActionRobot.intToByteArray(self.id))
        self.solved = np.zeros(info[2])
        self.checkMyScore()
        print "## current level : %d" % self.currentLevel

        # DDPG
        # random 하게 critic, actor, target critic net, target actor net 을 초기화하고
        # experience memory 도 deque 로 초기화 한다
        exploration_noise = OUNoise(
            self.num_actions, self.noise_mean, self.noise_sigma)
        counter = 1
        reward_per_episode = 0      # episode는 한 판을 의미.
        total_reward = 0
        print "# of States : ", self.num_states
        print "# of Actions : ", self.num_actions

        # reward 저장
        reward_st = np.array([0])

        # parameter 로 정한 episode 수 만큼 training 학습 진행
        for i in xrange(episodes):

            # 다음 레벨 받아오기
            self.currentLevel = self.getNextLevel()
            # 받아온 레벨이 1~3 이면 해당 레벨 로드, 아니면 1로 초기화 후 로드
            if self.currentLevel < 4:
                self.ar.loadLevel(self.currentLevel)
            else:
                self.currentLevel = 1
                self.ar.loadLevel(self.currentLevel)

            prevscore = 0
            reward_per_episode = 0
            steps = 0
            print "======== Starting Episode No : ", (i + 1), "========", "\n"

            # 하나의 episode 에 대한 루프
            while True:

                # 게임 플레이 screenshot 가져오기
                screenshot = self.ar.doScreenShot()
                x = self.getScreenBuffer(screenshot, self.width, self.height)
                # actor evaluation 을 통해서 다음에 취할 action 을 얻는다
                action = self.ddpg.evaluate_actor(np.reshape(
                    x, [1, self.num_states[0], self.num_states[1], self.num_states[2]]))
                print "## Get Action from network!! : ", action
                action = action[0]
                noise = exploration_noise.noise()
                # action 을 현재의 policy 에 따라 정하되,
                # epsilon(noise) 수치 정도에 따라 실험적인 action을
                # stochastic 하게 취하도록 한다.
                action = action + noise
                print action
                # distance 가 음수이면 양수로 뒤집어준다.
                action[0] = action[0] if action[0] > self.action_space_low[0] else -action[0]
                # distance 가 최대 범위를 넘어서면 최대 범위로 설정한다.
                action[0] = action[0] if action[0] < self.action_space_high[0] else self.action_space_high[0]
                # 각도의 경우에도 마찬가지 처리를 해준다
                action[1] = action[1] if action[1] > self.action_space_low[1] else -action[1]
                action[1] = action[1] if action[1] < self.action_space_high[1] else self.action_space_high[1]
                # tap time 도 마찬가지
                action[2] = action[2] if action[2] > self.action_space_low[2] else -action[2]
                action[2] = action[2] if action[2] < self.action_space_low[2] else self.action_space_high[2]
                print "## Action at step ", steps, " :", action, "\n"
                # 쏘고나서 점수가 안정화 될 때까지 조금 기다리는 로직이 들어있다
                state = self.shoot(action)

                if state == state.WON or state == state.LOST:
                    # episode 가 끝나면( 한 레벨이 끝나면 )
                    print "## Episode End"

                    screenshot = self.ar.doScreenShot()
                    observation = self.getScreenBuffer(
                        screenshot, self.width, self.height)

                    # 이기면 reward를 받고 지면 받지 않는다.
                    if state == state.WON:
                        score = self.se.getScoreEndGame(screenshot)
                        # 현재 step에서 얻은 점수를 1000으로 나눈 값을 reward 로 사용
                        # step 이란 새를 한 번 쏘는 행위를 뜻한다.
                        reward = (score - prevscore) / 1000.0
                    else:
                        reward = 0.00

                    self.currentLevel = self.currentLevel
                    self.firstShot = True   # episode 가 끝나면 first shot 초기화
                    done = True             # episode done 처리

                    # experience memory 에
                    # s(t), s(t + 1), action, reward 를 저장한다
                    print "######## SCORE : ", score
                    print "######## REWARD : ", reward
                    # x = state(screenBuffer) at t
                    # obervation = state(screenBuffer) at (t + 1)
                    self.ddpg.add_experience(
                        x, observation, action, reward, done)

                    # critic network 와 actor network 학습
                    # 정해둔 step 이상 진행됐을 경우부터 학습을 시작하도록 한다.
                    # experience 를 충분히 경험해야 하기 때문.
                    if counter > TRAIN_STEP:
                        self.ddpg.train()
                    counter += 1
                    steps += 1

                    print "==== EPISODE: ", i, ' Steps: ', steps, ' Total Reward: ', reward_per_episode
                    print "Writing reward info into file..."
                    exploration_noise.reset()
                    # reward_st 는 배열이다.
                    # 마지막 원소에 해당 판에서 얻은 총 점수를 기록하고
                    # 파일로 내보낸다
                    reward_st = np.append(reward_st, reward_per_episode)
                    np.savetxt("episodes_reward.txt", reward_st, newline="\n")
                    print "\n\n"

                    break

                elif state == state.PLAYING:    # PLAYING 상태일 때
                    screenshot = self.ar.doScreenShot()
                    vision = Vision(screenshot)
                    sling = vision.findSlingshotMBR()

                    while sling == None and self.ar.checkState() == state.PLAYING:
                        print "## No slingshot was detected. Please remove pop up or zoom out"
                        self.ar.fullyZoomOut()
                        screenshot = self.ar.doScreenShot()

                    # S(t + 1) 을 얻는다
                    observation = self.getScreenBuffer(
                        screenshot, self.width, self.height)
                    # experience memory 에
                    # S(t), S(t + 1), action, reward 를 저장한다
                    score = self.ar.getInGameScore(screenshot)
                    reward = (score - prevscore) / 1000.0
                    prevscore = score
                    done = False
                    reward_st = np.append(reward_st, reward)

                    self.ddpg.add_experience(
                        x, observation, action, reward, done)
                    print "## Add experience (action) (reward) (done)", action, reward, done

                    # critie, actor network 학습
                    if counter > TRAIN_STEP:
                        self.ddpg.train()
                    reward_per_episode += reward
                    counter += 1
                    steps += 1

                # 일반적인 상황이 아닌 상황들에 대한 예외처리
                elif state == state.LEVEL_SELECTION:
                    print "unexpected level selection page, go to the last current level: %d" % self.currentLevel
                    self.ar.loadLevel(self.currentLevel)
                elif state == state.MAIN_MENU:
                    print"unexpected main menu page, reload the level: %d" % self.currentLevel
                    self.ar.loadLevel(self.currentLevel)
                elif state == state.EPISODE_MENU:
                    print "unexpected episode menu page, reload the level: %d" % self.currentLevel
                    self.ar.loadLevel(self.currentLevel)

        total_reward += reward_per_episode  # episode 들의 reward 를 누계
        avg_reward = total_reward / episodes
        print "## Average reward per episode is : ", avg_reward


if __name__ == "__main__":

    PlayAgent = PlayAgent()
    PlayAgent.ddpg_run()
    shutdownJVM()
