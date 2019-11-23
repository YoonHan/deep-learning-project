# -*- coding: utf-8 -*
"""
    기존에 Java로 작성된 게임플레이 agent 를 파이썬 버전으로 포팅.
"""
from jpype import *
import os
import numpy as np
import sys
import time
import threading
from collections import deque


# Jpype 세팅용 변수들.
# Jpype: python 에서 Java class library 에 접근할 수 있도록 해주는 패키지
jarpath = os.path.join(os.path.abspath('../'))    # external jar path
classpath = os.path.abspath('../output')
# JVM 돌릴 때 jarpath 전달
print(getDefaultJVMPath())
startJVM(getDefaultJVMPath(),
         "-Djava.ext.dirs=%s -Djava.class.path=%s" % (jarpath, classpath))

# Angry Bird source package 로드
TrajectoryPlannerPkg = JPackage('ab').planner
DemoOtherPkg = JPackage('ab').demo.other
VisionPkg = JPackage('ab').vision
AwtPkg = JPackage('java').awt
UtilPkg = JPackage('java').util

# Agent 구성에 필요한 클래스 로드
TrajectoryPlanner = TrajectoryPlannerPkg.TrajectoryPlanner
ClientActionRobot = DemoOtherPkg.ClientActionRobot
ClientActionRobotJava = DemoOtherPkg.ClientActionRobotJava
ABObject = VisionPkg.ABObject
GameState = VisionPkg.GameStateExtractor
Vision = VisionPkg.Vision

# 필요한 parameter 들 선언
episodes = 200      # 반복 횟수
TRAIN_STEP = 30
BATCH_SIZE = 20

ALL_WIN_LEVEL = 20  # 이 레벨까지 클리어하면 agent 종료
TARGET_SCORE = 999999   # agent 의 목표 점수
TEST_FLAG = True
START_WITH_TEST = False

SMALL_NOISE_LEVEL = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
TRAIN_LEVEL = [1, 2, 3, 5, 8, 13, 15, 17, 19, 20]


class RLAgent:      # Agent class
    def __init__(self, ip="127.0.0.1", id=9000):
        """
            RLAgent constructor.
            ip 는 서버 ip 이고,
            id 는 포트 번호이다.
        """
        f = open("training_loss.txt", 'a')
        data = "Batch size: %d" % BATCH_SIZE
        f.write(data)
        f.close()

        self.ar = ClientActionRobotJava(ip)
        self.tp = TrajectoryPlanner()    # 새의 궤적을 계산해주는 instance
        self.firstShot = True
        self.solved = []
        self.currentLevel = -1
        self.failedCounter = 0
        self.id = id
        self.width = 840
        self.height = 480
        # Raw image size = [width, height, 3(RGB)]
        self.num_states = 400
        # Action space 정의 : [거리는 0부터 45까지, 각도는 0도부터 90도까지, taptime 은 0부터 5000ms 까지]
        self.num_actions = 3
        self.action_space_high = [45, 45, 1500]
        self.action_space_low = [40, 10, 1000]
        self.noise_mean = [0, 0, 0]     # noise 평균
        self.noise_sigma = [2, 10, 200]  # noise 표준편차
        self.noise_sigma_small = [1, 1, 0]
        self.ddpg = DDPG(self.num_states, self.num_actions,
                         self.action_space_high, self.action_space_low)
        self.final_observation = 0
        self.Final_observation_error = False
        self.temp_replay_memory = deque()

        self.total_numbirds = [3, 5, 4, 4, 4,
                               4, 4, 4, 4, 5,
                               4, 4, 4, 4, 4,
                               5, 3, 5, 4, 5,
                               8]  # level 마다의 새의 총 마리 수
        self.steps = 0  # training step
        self.test_flag = START_WITH_TEST
        self.all_win_flag = True    # 모든 level 을 클리어해야 끝남
        self.test_totalscore = 0.0  # test 수행 시에 총 점수 저장

        self.TEST_START_FLAG = False
        # threading 으로 CLI instance 를 병렬적으로 동작시킴.
        t1 = threading.Thread(target=self.CLI)
        t1.start()
        self.TEST_START_LEVEL = 1   # test 시작 level
        self.TRAIN_START_LEVEL = TRAIN_LEVEL[0]  # train 시작 level
        self.TRAIN_START_FLAG = False
        self.TEST_AFTER_EVERY_EPISODE = False   # 매 episode 가 끝날 때마다 test 수행 여부 결정
        self.TEMP_TEST_AFTER_EVERY_EPISODE = False  # 임시변수

    def getNextLevel(self):
        """
            다음 level로 이동한다.
        """
        level = 0
        unsolved = False    # 해당 레벨이 클리어됐는지 여부

        for i in range(len(self.solved)):
            if self.solved[i] == 0:
                unsolved = True
                level = i + 1   # level 은 list index + 1

                # 만약 unsolved 인 level 이 현재 level 보다 작고 현재 level 이 solved 된 level 개수보다 적다면,
                # 다음 인덱스로 넘어간다
                if level <= self.currentLevel and self.currentLevel < len(self.solved):
                    continue
                else:       # 아니면 해당 level 을 반환
                    return level

        if unsolved:
            return level

        level = (self.currentLevel + 1) % len(self.solved)
        if level == 0:
            level = len(self.solved)

        return level

    def checkMyScore(self):
        scores = self.ar.checkMyScore()
        level = 1
        for s in scores:
            print(f"level {level}: {s}")
            if s > 0:
                self.solved[level - 1] = 1
            level += 1

    def getScreenBuffer(self, buffer, width=840, height=400):
        """
            게임 화면 정보를 가져온다
        """
        print("GET SCREEN BUFFER\n")
        returnBuffer = np.zeros((3, height, width))
        for i in range(height):
            for j in range(width):
                RGB = buffer.getRGB(j, i)   # 각 pixel 의 RGB 정보
                returnBuffer[0, i, j] = RGB & 0x0000ff
                returnBuffer[1, i, j] = RGB & 0x00ff00
                returnBuffer[2, i, j] = RGB & 0xff0000

        print("RETURN SCREEN BUFFER\n")
        return returnBuffer

    def shoot(self, action):
        """
            action 에 따라 새를 쏜다.
        """
        screenshot = self.ar.doScreenShot()
        vision = Vision
        sling = vision.findSlingshotMBR()
        state = self.ar.checkState()

        while sling == None and state == state.PLAYING:
            print("no slingshot detected. Please remove pop up or zoom out\n")
            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()

        pigs = vision.findPigsMBR()
        state = self.ar.checkState()

        # if there is a sling, then play, otherwise skip
        if sling != None:
            refPoint = self.tp.getReferencePoint(sling)

            # ddpg 인스턴스로부터 슈팅 지점 정보를 받아온다.
            # tapTime 은 ms 단위
            releaseDistance = action[0]  # 슈팅할 거리
            releaseAngle = action[1] * 100  # 각도
            tapTime = action[2]  # tap time

            print(f"Release Distance: {releaseDistance}")
            print(f"Release Angle: {releaseAngle / 100}")

            self.ar.fullyZoomOut()
            prev_screenshot = self.ar.doScreenShot()
            prev_vision = self.Vision(prev_screentshot)
            _sling = prev_vision.findSlingshotMBR()

            if self.currentLevel < 10:
                prev_num_birds = (prev_vision.findBirdsMBR()).size()

                if prev_num_birds == 0:
                    time.sleep(5)
                    state = self.ar.checkState()
                    return state

            if _sling != None:
                scale_diff = (sling.width - _sling.width) ** 2 + \
                    (sling.height - _sling.height) ** 2
                if scale_diff < 25:
                    if self.test_flag:  # test 환경에서는 shooting time 을 저장하지 않는다.
                        self.ar.shoot(int(refPoint.x), int(refPoint.y), int(
                            releaseDistance), int(releaseAngle), 0, int(tapTime), True)
                        time.sleep(1)
                        state = self.ar.checkState()
                    elif self.currentLevel > 9:
                        self.ar.fastshoot(int(refPoint.x), int(refPoint.y),
                                          -int(releaseDistance *
                                               cos(radians(releaseAngle / 100))),
                                          int(releaseDistance *
                                              sin(radians(releaseAngle / 100))), 0, int(tapTime))
                        shoot_time = time.time()
                        temp_observation = 0
                        time.sleep(3)
                        while time.time() - shoot_time < 15:
                            screenshot = self.ar.doScreenShot()
                            state = self.ar.checkState()
                            if state == state.WON or state == state.LOST:
                                break
                            vision = Vision(screenshot)
                            temp_observation = self.FeatureExtractor(vision)

                        if state == state.WON or state == state.LOST:
                            self.final_observation = temp_observation
                            self.Final_observation_error = False
                            time.sleep(3)   # 점수가 표시될 때까지 기다림.
                    else:
                        shoot_time = time.time()
                        self.ar.fastshoot(int(refPoint.x), int(refPoint.y), -int(releaseDistance * cos(radians(
                            releaseAngle / 100))), int(releaseDistance * sin(radians(releaseAngle / 100))), 0, int(tapTime), False)
                        time.sleep(1)
                        self.ar.fullyZoomOut()
                        screenshot = self.ar.doScreenShot()
                        vision = Vision(screenshot)
                        curr_num_birds = (vision.findBirdsMBR()).size()
                        curr_num_pigs = (vision.findPigsMBR()).size()

                        final_observation_flag = False
                        # 새를 다 쏠 때까지 반복
                        while curr_num_birds != prev_num_birds - 1:
                            state = self.ar.checkState()
                            if state == state.WON or state == state.LOST:
                                print("break!!!")
                                break
                            self.ar.fullyZoomOut()
                            screenshot = self.ar.doScreenShot()
                            vision = Vision(screenshot)
                            curr_num_birds = (vision.findBirdsMBR()).size()
                            curr_num_pigs = (vision.findPigsMBR()).size()

                            # 30초 이상 지나면 루프를 탈출
                            if time.time() - shoot_time > 30:
                                break

                            # 모든 pig가 제거되면 final observation 업데이트
                            if curr_num_pigs == 0 or curr_num_birds == 0:
                                if curr_num_birds == prev_num_birds - 1:
                                    print("Final observation made")
                                    self.final_observation = self.FeatureExtractor(
                                        vision)
                                    self.final_observation[400] = prev_num_birds - 1
                                    final_observation_flag = True
                                    self.Final_observation_error = False
                                if curr_num_birds == prev_num_birds and final_observation_flag == False:
                                    print("Temporary final obersvation made")
                                    self.final_observation = self.FeatureExtractor(
                                        vision)
                                    self.Final_observation[400] = prev_num_birds - 1
                                    final_observation_flag = True
                                    self.Final_observation_error = False

                        if curr_num_pigs == 0 or curr_num_birds == 0:
                            if final_observation_flag == False:
                                # level 이 정상적으로 마무리 되지 않았다면 해당 experience 를 버린다.
                                print(
                                    "Final observaiton capturing error. This experience will be dropped")
                                self.Final_observation_error = True
                            while state != state.WON and state != state.LOST:
                                state = self.ar.checkState()
                                if time.time() - shoot_time > 30:
                                    break
                            time.sleep(3)   # score 가 뜰 때까지 기다림

                    if state == state.PLAYING:
                        self.firstShot = False
                else:
                    print(
                        "Scale is changed. So cannot execute the shot, will resegment the image")
            else:
                print(
                    "no sling obejct is detected. cannot execute the shot, will resegment the image")
        return state

    def ddpg_run(self):
        """
            run ddpg algorithm with raw pixel information
        """
        info = self.ar.configure(ClientActionRobot.intToByteArray(self.id))
        self.solved = np.zeros(info[2])

        TRAIN_LEVEL_index = 0
        if TEST_FLAG and START_WITH_TEST:
            self.currentLevel = 1
        else:
            self.currentLevel = TRAIN_LEVEL[TRAIN_LEVEL_index]

        # DDPG 알고리즘
        # critic, actor, target critic network, target actor network 와 replay buffer 를 램덤하게 초기화
        exploration_noise = OUNoise(
            self.num_actions, self.noise_mean, self.noise_sigma)
        exploration_noise_small = OUNoise(
            self.num_actions, self.noise_mean, self.noise_sigma_small)
        counter = 0
        reward_per_episode = 0
        total_reward = 0
        print(f"Number of States: {self.num_states}")
        print(f"Number of Actions: {self.num_actions}")

        # 동일한 level 에 대해서 training 한다.
        # training 횟수는 미리 정해진 만큼(200 회 정도)
        for current_episode in xrange(episodes):
            if counter > TRAIN_STEP:
                self.TEST_AFTER_EVERY_EPISODE = self.TEMP_TEST_AFTER_EVERY_EPISODE
            self.ar.loadLevel(self.currentLevel)
            print(f"===== Starting episode no: {current_episode} =====")
            self.ar.fullyZoomOut()
            screenshot = self.ar.doScreenShot()
            vision = Vision(screenshot)
            sling = vision.findSlingshotMBR()
            state = self.ar.checkState()
            while sling == None and state == state.PLAYING:
                print("no slingshot detected. Please remove pop up or zoom out")
                self.ar.fullyZoomOut()
                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                sling = vision.findSlingshotMBR()

            self.steps = 0
            observation = self.FeatureExtractor(vision)

            prev_score = 0
            reward_per_episode = 0
            self.final_observation = 0
            self.Final_observation_error = True

            # 하나의 episode 에 대해서
            while True:
                if self.test_flag:
                    print(
                        "\n###############################################################")
                    print(
                        "############                 TEST                  ############")
                    print(
                        "###############################################################\n")
                else:
                    print(
                        "\n###############################################################")
                    print(
                        "############                 TRAIN                 ############")
                    print(
                        "###############################################################\n")
                print(f"current level: {self.currentLevel}")

                x = observation
                action = self.ddpg.evaluate_actor(
                    np.reshape(x, [1, self.num_states]))
                print(f"Action from network: {action}")
                action = action[0]
                noise = exploration_noise.noise()
                for l in SMALL_NOISE_LEVEL:
                    if self.currentLevel == l:
                        noise = exploration_noise_small.noise()
                        break

                if self.test_flag == False:
                    # 취할 action 을 현재의 policy 와 exploration noise epsilon 에 따라서 결정한다
                    action = action + noise
                print(action)
                action[0] = action[0] if action[0] > self.action_space_low[0] else 2 * \
                    self.action_space_low[0] - action[0]
                action[0] = action[0] if action[0] < self.action_space_high[0] else self.action_space_high[0]
                action[1] = action[1] if action[1] > self.action_space_low[1] else 2 * \
                    self.action_space_low[1] - action[1]
                action[1] = action[1] if action[1] < self.action_space_low[1] else self.action_space_high[1]
                action[2] = action[2] if action[2] > self.action_space_low[2] else 2 * \
                    self.action_space_low[2] - action[2]
                action[2] = action[2] if action[2] < self.action_space_high[2] else self.action_space_high[2]
                print(f"Action at step {self.steps}: {action}")

                screenshot = self.ar.doScreenShot()
                vision = Vision(screenshot)
                prev_num_birds = (vision.findBirdsMBR()).size()

                state = self.shoot(action)

                if state == state.WON or state == state.LOST:
                    # episode 가 끝났을 경우,
                    print("End of Game")
                    screenshot = self.ar.doScreenShot()

                    if state == state.WON:
                        score = self.ar.getScoreEndGame(screenshot)
                        reward = (score - prev_score) / 1000.0
                    else:
                        reward = -100.00

                    # Writing TEST result
                    if self.test_flag and state == state.WON:
                        self.test_totalscore += score
                        f = open("Test_Result.txt", 'a')
                        data = "%d Win %d\b" % (self.currentLevel, score)
                        f.write(data)
                        f.close()

                    # WIN/LOSE 에서 얻은 reward를 forward propagate 시킨다.
                    if self.steps != 0:
                        temp_reward = reward * 0.5
                        temp = self.temp_replay_memory.pop()
                        temp_x = temp[0]
                        temp_x = np.reshape(temp_x, [self.num_states])
                        temp_observation = temp[1]
                        temp_observation = np.reshape(
                            temp_observation, [self.num_states])
                        temp_action = temp[2]
                        temp_action = np.reshape(
                            temp_action, [self.num_actions])
                        while True:
                            try:
                                if self.test_flag == False:
                                    self.ddpg.add_experience(
                                        temp_x, temp_observation, temp_action, temp[3] + temp_reward, temp[4])
                                    temp = self.temp_replay_memory.pop()
                                    temp_x = temp[0]
                                    temp_x = np.reshape(
                                        temp_x, [self.num_states])
                                    temp_observation = temp[1]
                                    temp_observation = np.reshape(
                                        temp_observation, [self.num_states])
                                    temp_action = temp[2]
                                    temp_action = np.reshape(
                                        temp_action, [self.num_actions])
                                    temp_reward = temp_reward * 0.5
                            except:
                                break

                    # 마지막 episode 까지 돌았으니 다음 level 로 간다.
                    if self.test_flag:  # TEST 인 경우
                        if self.currentLevel == ALL_WIN_LEVEL:
                            f = open("Test_Result.txt", 'a')
                            data = "All levels cleared. Stopping game\n Total Score: %d\n" % (
                                self.test_totalscore)
                            f.write(data)
                            f.close()
                            self.ddpg.save_parameters()
                            return -1
                        self.currentLevel = self.getNextLevel()
                    else:   # TRAINING 인 경우
                        if TRAIN_LEVEL_index == len(TRAIN_LEVEL) - 1 or self.TEST_START_FLAG or self.TEST_AFTER_EVERY_EPISODE:
                            if TEST_FLAG:
                                if self.TEST_START_FLAG:
                                    self.currentLevel = self.TEST_START_LEVEL
                                else:
                                    self.currentLevel = 1
                                self.TEST_START_FLAG = False
                                self.ddpg.save_parameters()
                                self.ddpg.restore_parameters()
                                self.test_flag = True
                                self.all_win_flag = True
                                self.test_totalscore = 0.0
                                f = open("Test_Result.txt", 'a')
                                data = "Test at episode: %d\n" % current_episode
                                f.write(data)
                                f.close()
                            else:   # TEST가 아니면 첫 번째 TRAINING LEVEL 부터 시작
                                self.currentLevel = TRAIN_LEVEL[0]
                        else:
                            TRAIN_LEVEL_index += 1
                            self.currentLevel = TRAIN_LEVEL[TRAIN_LEVEL_index]

                    # 게임의 LOST state 일 때 test를 중단한다
                    if self.test_flag and (self.currentLevel != 1 and self.currentLevel != self.TEST_START_LEVEL):
                        if state == state.LOST or self.TRAIN_START_FLAG:
                            f = open("Test_Result.txt", 'a')
                            if self.TEST_START_FLAG:
                                f.write("Stopped by user. LOSE")
                            data = "Total Score: %d\b" % self.test_totalscore
                            f.write(data)
                            f.close()

                            if TARGET_SCORE < self.test_totalscore:
                                f = open("Test_Result.txt", 'a')
                                data = "Exceed Target Score. Stop game\nTotal Score: %d\n" % self.test_totalscore
                                f.write(data)
                                f.close()
                                self.ddpg.save_parameters()  # 파라미터 저장
                                return -1   # game is in end state

                            self.test_flag = False

                            if self.TEST_AFTER_EVERY_EPISODE:
                                TRAIN_LEVEL_index += 1
                                if TRAIN_LEVEL_index == len(TRAIN_LEVEL):
                                    TRAIN_LEVEL_index = 0
                            else:
                                TRAIN_LEVEL_index = 0

                            if self.TEST_START_FLAG or self.TEST_AFTER_EVERY_EPISODE:
                                self.currentLevel = self.TRAIN_START_LEVEL
                            else:
                                self.currentLevel = TRAIN_LEVEL[TRAIN_LEVEL_index]
                            self.TRAIN_START_FLAG = False

                    self.firstShot = True
                    done = True
                    # s(t), s(t + 1), action, reward 를 메모리에 저장한다
                    print(f"######### SCORE {score}")
                    print(f"######### REWARD {reward}")
                    if self.Final_observation_error == False:
                        print(
                            f"######## Add experience {action} {reward} {done}")
                        if self.test_flag == False:
                            self.ddpg.add_experience(
                                x, observation, action, reward, done)
                    elif self.test_flag:
                        print()
                    else:
                        print(
                            "######## Not adding experience because of capturing ERROR")

                    # critic network 와 actor network 를 training 시킨다
                    if counter > TRAIN_STEP:
                        if self.test_flag == False:
                            self.ddpg.train()
                    reward_per_episode += reward
                    if self.test_flag == False:
                        counter += 1
                    self.steps += 1

                    # episode 가 끝났는지 확인
                    print(
                        f"EPISODE: {current_episode} Steps: {self.steps} Total Reward: {reward_per_episode}")
                    exploration_noise.reset()
                    exploration_noise_small.reset()
                    print("\n\n\n")

                    break

                elif state == state.PLAYING:
                    # 게임 state 가 PLAYING 상태이면
                    # Object detection 을 하고
                    self.ar.fullyZoomOut()
                    screenshot = self.ar.doScreenShot()
                    vision = Vision(screenshot)
                    sling = vision.findSlingshotMBR()
                    state = self.ar.checkState()
                    while sling == None and state == state.PLAYING:
                        print(
                            "no slingshot detected. Please remove pop up or zoom out")
                        self.ar.fullyZoomOut()
                        screenshot = self.ar.doScreenShot()
                        vision = Vision(screenshot)
                        sling = vision.findSlingshotMBR()

                    self.steps += 1

                    # s(t + 1) 상태(게임 스크린샷)를 얻는다
                    vision = Vision(screenshot)
                    print("Next state is captured")
                    observation = self.FeatureExtractor(vision)
                    observation[400] = prev_num_birds - 1

                    # experience memory 에 s(t) s(t + 1), action, reward 를 저장한다
                    score = self.ar.getInGameScore(screenshot)
                    reward = (score - prev_score) / 1000.0
                    if reward == 0.0:
                        reward = -100.0

                    print(f"######## instant reward {reward}")
                    prev_score = score
                    done = False

                    self.temp_replay_memory.append(
                        (x, observation, action, reward, done))
                    if self.test_flag == False:
                        print(
                            f"######## add experience {action} {reward} {done}")

                    # ciritic network 와 actor network 를 training 시킨다
                    if counter > TRAIN_STEP:
                        if self.test_flag == False:
                            self.ddpg.train()
                    reward_per_episode += reward
                    if self.test_flag == False:
                        counter += 1

                    exploration_noise.reset()
                    exploration_noise_small.reset()

                # 예외적인 상황들 처리
                elif state == state.LEVEL_SELECTION:
                    print(
                        f"unexpected level selection page. go to the recent level: {self.currentLevel}")
                    self.ar.loadLevel(self.currentLevel)
                elif state == state.MAIN_MENU:
                    print(
                        f"unexpected main menu page. go to the recent level: {self.currentLevel}")
                    self.ar.loadLevel(self.currentLevel)
                elif state == state.EPISODE_MENU:
                    print(
                        f"unexpected episode menu page. go to the recent level: {self.currentLevel}")
                    self.ar.loadLevel(self.currentLevel)

        total_reward += reward_per_episode
        print(f"Average reward per episode {total_reward / episode}")

    def FeatureExtractor(self, vision):
        """
            Vision 모듈에서 DDPG 에 필요한 observation 을 얻어온다
        """

        # observation 은 다음과 같은 정보를 가지고 있다.
        # 400차원의 벡터인 Grid. 20 X 20 크기이고 각 grid cell이 정보에 따라 다음 수치들을 부여한다.
        # 아무것도 없음: 0
        # pig 만 있음: 10
        # 장애물만 있음: -10
        # pig 와 장애물(block object)이 있음: 5

        # width position = 440 ~ 840 이고 height position = 240~640 범위이므로 20 X 20 의 tile 로 나눈다면,
        # index = (width-440)/20 + ((height-240)/20) * 20
        observation = np.zeros((self.num_states), dtype=np.float32)

        pigs = vision.findPigsMBR()
        for i in xrange(pigs.size()):
            temp_object = pigs.get(i)
            # detection 된 object의 center position 좌표를 구한다
            center_x = temp_object.x + temp_object.width / 2
            center_y = temp_object.y + temp_object.height / 2

            if center_x - 440 > 0 and center_y - 240 > 0:
                observation[(int)((center_y-440)/20)*20 +
                            (int)((center_x-240)/20)] = 10

        blocks = vision.findBlocksMBR()
        for i in xrange(block.size()):
            temp_object = blocks.get(i)
            center_x = temp_object.x + temp_object.width / 2
            center_y = temp_object.y + temp_object.height / 2

            if center_x - 440 > 0 and center_y - 240 > 0:
                # pig 가 grid 영역안에 존재하지 않을 때
                if observation[(int)((center_y-440)/20)*20 + (int)((center_x-240)/20)] == 0:
                    observation[(int)((center_y-440)/20)*20 +
                                (int)((center_x-240)/20)] = -10
                else:
                    observation[(int)((center_y-440)/20)*20 +
                                (int)((center_x-240)/20)] = 5

        # 현재 레벨에서 총 남은 새의 숫자를 observation 에 기록
        # 현재 레벨 숫자도 기록
        observation[400] = self.total_numbirds[self.currentLevel - 1] - self.steps
        observation[401] = self.currentLevel

        return observation

    def CLI(self):
        """
            표준 입력으로부터 입력값을 받아서
            test 할지 train 할지에 대한 분기처리를 함
        """
        while True:
            line = sys.stdin.readline()
            try:
                if line.split(' ')[0] == "set" and line.split(' ')[1] == "test":
                    setlevel = int(line.split(' ')[2])
                    print(f"Set test start level to {setlevel}")
                    self.TEST_START_LEVEL = setlevel
                if line.split(' ')[0] == "test" and line.split(' ')[1] == "train":
                    setlevel = int(line.split(' ')[2])
                    print(f"Set train start level to {setlevel}")

                if line == "test\n":
                    self.TEST_START_FLAG = True
                    print("Test start flag is set")
                if line == "train\n":
                    self.TEST_START_FLAG = False
                    self.TRAIN_START_FLAG = True
                    print("Train start flag is set")
                if line == "TEST_EVERY\n":
                    print("Test start after training one episode")
                    self.TEMP_TEST_AFTER_EVERY_EPISODE = True
                if line == "TEST_CYCLE\n":
                    print("Test start after training cycle")
                    self.TEMP_TEST_AFTER_EVERY_EPISODE = False
            except:
                print("ERROR!!")


if __name__ == '__main__':
    PlayAgent = RLAgent()
    PlayAgent.ddpg_run()
    print("main called")
    shutdownJVM()  # JVM 을 shutdown 해주지 않으면, 다음번 JVM load 시에 문제가 발생한다.
