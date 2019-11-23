# -*- encoding: utf-8 -*-
from jpype import *
import os

jarpath = os.path.join(os.path.abspath('../external'))    # external jar path
startJVM(getDefaultJVMPath(),
         "-Djava.ext.dirs=%s" % jarpath)    # JVM 돌릴 때 jarpath 전달

# Angry Bird source package 로드
TrajectoryPlannerPkg = JPackage('ab').planner
DemoOtherPkg = JPackage('ab').demo.other
VisionPkg = JPackage('ab').vision
AwtPkg = JPackage('java').awt
UtilPkg = JPackage('java').util

# Java 내장 class 로드
# Point = AwtPkg.Point
print(AwtPkg)
Rectangle = AwtPkg.Rectangle

print "done"
