import re

DIRECTION = [
    "left", "right", "behind", "front", "end", "rear", "back", "between", "before",
    "rightward", "leftward", "backward", "backwards",
    "south", "north", "east", "south", "west", "northwest", "northeast", "southwest", "southeast"
]

ACTION = [
    "circle", "buttonhook", "around", "round"
]

QUANTITY = [
    "45", "90", "145", "180", "270", "degrees", "half", "quarter"
]

REJECT = [
    "robot", "side", "lrb", "rrb", "head"
]

# ADJ
COLOR = [
    "red", "white", "yellow", "pink", "blue", "gold", "purple"
]

NOUN = [
    "well", "mushroom", "windmill", "stump", "stone", "rock", "plant", "pond", "bear",
    "post", "booth", "chair", "chest", "column", "fence", "fern", "ladder",
]

SPECIAL_WORDS = [
    "it"
]

DIRECTION_DEF = [
    "{landmark} on {direction}",  # apple on your left -> left of apple
    "{landmark} from {direction}",  # the tree from the right
    "{direction} of {landmark}",  # left side of apple
]

BE_CAREFUL = [
    "right hand side",
    "pass on the right hand side",
    "turn right and move to the left of tree",  # --> right left tree
    "the flower on your left and the white fence on your right"
]

NOUN_OR_IT = [
    "stop",
]

I_DONT_KNOW = [
    "alongside", "beside"
]

ERROR_WORDS = [
    "betweent", "betwen",
]


HOW_TO_DEAL_WITH = [
    "move forward until you reach 1 vehicle width away from the red fence",
]

# --> (action, object, direction)
RULES = [
    [r"as well as (pass) (the cylinder shaped building) and turn (left) after you pass it", (1,2,None),(3,None,None)],
    [r"hug the (lake) to your (left) until you (get) the (cactus)", ("cross left", 1, None),(3, 4, None)],
    [r"keeping the cactus to your right   go behind it   turning in the yellow fence direction",(None,"the cactus","left behind")],
    [r"take half a circle around (.{1,20}) (right|left) side", ("half a circle",1,2)],
    [r"3   walk around windmill right side and five steps around the pink flower pot right side",("around", "windmill","right"),("five steps around", "pink flower pot","right")],
    [r"and go around on the right",("around", "box", "left")],
    [r"fly between the (green toy soldier) and (flowers)",("between",12,None)],
    [r"cross over them move straight",("cross", "apple and drum", None)],
    [r"and turn around it and stop",("around","orange cone", None)],
    [r"with slight curve over there move towards wall over near cut down tree",(None,"cut down tree",None)],
    [r"take around over cut down tree",("around","cut down tree", None)],
    [r"take a slight bend over near well",("take a slight bend","well",None)],
    [r"move in between well and car towards apple",(None,"apple",None)],
    [r"fly between the (darker colored wood barrel) and (red fence)",("between",12,None)],


    [r"keep going straight",(None,None,None)],
    [r"walk (around) the (.{1,20}) on the side (facing|facing away from) the (.{1,20})", (3,4,None),(1,2,None)],
    [r"(until|till|til) you reach the other end$",("reach","the other end",None)],
    [r"(until|till|til) you (reach|reaching|reach near|get to|see|get|see|hit) (the|a) (.{1,20})$",(None,4,None)],
    [r"(until|till|til) you (reach|reaching|reach near|get to|see|get|hit) (.{1,20})$",(None,3,None)],
    [r"(until|till|til) (reach|reaching|reach near|get to|see|get) (the|a) (.{1,20})$",(None,4,None)],
    [r"(reach|reach near|reaching|proceed to|stop at|get) (the|a) (.{1,20})($|\s)",(None,3,None)],
    [r"(past|around|pass|passing) (the|a) (.{1,20}) leaving it on (the|your) (right|left)",(1,3,40)],
    [r"(go|turn) (left|right) (pass|past|passing) (.{1,20}) until (you facing|you face|face|facing) (.{1,20})$",(2,4,None),("face",6,None)],
    [r"(move|go|turn|curve|circle|walk) (around|round|right around|left around|right round|left round) (the|a) (.{1,20}) (then|and|\s)",(2,4,None)],
    [r"(move|go|turn|curve|circle|walk) (around|round|right around|left around|right round|left round) (the|a) (.{1,20}) (then|and|\s)",(2,4,None)],
    [r"(move|go|turn|curve|circle|walk) (around|round) (the|a) (.{1,20}) on (the|your) (left|right|left hand|right hand) side",("around",4,5)],
    [r"(past|around|pass|passing) (the|a) (.{1,20}) on (the|your) (right|left)",(1,3,50)],
    [r"(go|turn|take a|make a) (left|right|left turn|right turn) and go (to|towards|toward) (the|a) (.{1,20}) (then|and)",(2,5,None)],
    [r"(go|turn|take a|make a) (left|right|left turn|right turn) go (to|towards|toward) (the|a) (.{1,20}) (then|and)",(2,5,None)],
    [r"(go|turn|take a|make a) (left|right|left turn|right turn) (to|towards|toward) (the|a) (.{1,20}) (then|and|closest)",(2,5,None)],
    [r"(go|turn|take a|make a) (left|right|left turn|right turn) (around|round) (the|a) (.{1,20})$",(2,5,None)],
    [r"(go|staying to the) (left|right) of the (.{1,20}) (then|and|closest)",(None,3,2)],
    [r"(go|staying to the) (left|right) of the (.{1,20})$",(None,3,2)],
    [r"(move|go|turn|curve|circle) (around|round|right around|left around|right round|left round) (pass|past|passing) the (.{1,20}) (then|and|\s)",(2,4,None)],
    [r"(go|turn|take a|make a) (left|right|left turn|right turn) (to|towards|toward) the (.{1,20}) (then|and|closest)",(2,4,None)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward|forward towards|straight towards|foward go to|straight for) the (right|left|right side|left side|right hand side|left hand side) of (.{1,20})(and|then|$)",(None,4,3)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward|forward towards|straight towards|foward go to|straight for) (right|left|right side|left side|right hand side|left hand side) of (.{1,20}\s)",(None,4,3)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward|forward towards|straight towards|foward go to|straight for) the (red and white propane tank)",(None,3,None)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward|forward towards|straight towards|foward go to|straight for) (.{1,20})$",(None,3,None)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward|forward towards|straight towards|foward go to|straight for) the (.{1,20}) (and|then|crossing|passing|so|by|\s)",(None,3,None)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward|forward towards|straight towards|foward go to|straight for) until you (pass|passing) the (.{1,20}) on your (right|left)",(3,4,50)],
    [r"(drive|move|go|head|heading|walk|fly) (behind|around|round) the (.{1,20})$",(2,3,None)],
    [r"(drive|move|go|head|heading|walk|fly) (behind|around|round) (.{1,20}) (left|right) side",(2,3,None)],
    [r"(drive|move|go|head|heading|walk|fly) (behind|around|round) (.{1,20})$",(2,3,None)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward) (the|a) (.{1,20})$",(None,4,None)],
    [r"(drive|move|go|head|heading|walk|fly) (to|toward|towards|forward) (.{1,20})$",(None,3,None)],
    [r"(turn|take a) (left|right) and go (to|towards|toward) (the|a) (.{1,20})$",(2,5,None)],
    [r"(turn|take a) (left|right) go (to|towards|toward) (the|a) (.{1,20})$",(2,5,None)],

    # between
    [r"between (the skinny and fat tower)",("between",1,None)],
    [r"between the (.{1,20})  lrb  on (your|the) (right|left)  rrb  and the (.{1,20})  lrb  on (your|the) (right|left)  rrb",("between",14,None)],
    [r"between (the|a) (.{1,20}) and (the|a) (.{1,20}) and then",("between",24,None)],
    [r"between the (.{1,20}) on (your|the) (right|left) and the (.{1,20}) on (your|the) (right|left)",("between",14,None)],
    [r"between (the|a) (.{1,20}) and (the|a) (.{1,20}) (turn|towards|toward|stop|until|and|\s)",("between",24,None)],
    [r"between (the|a) (.{1,20}) and (the|a) (.{1,20})",("between",24,None)],
    [r"between (.{1,20}) and (.{1,20}) (\s\s|stop|and continue)",("between",12,None)],
    [r"between (.{1,20}) and (.{1,20})",("between",12,None)],
    [r"(the|a) (.{1,20}) (at|on|to) (your|the) (right|left) and (the|a) (.{1,20}) (at|on|to) (your|the) (right|left)",(None,27,None)],
    [r"(go|gone|go straight) (pass|past) (the|a) (.{1,20}) to the (left|right)",(None,4,5)],
    [r"(go|gone|go straight) (pass|past) (the|a) (.{1,20}) to the (.{1,20})$",(None,5,None)],
    [r"(go|gone|go straight) (pass|past) (the|a) (.{1,20})",(None,4,None)],
    [r"turn (toward|towards|to) (the|a) (.{1,20})$",(None,3,None)],
    [r"to the (right|left) of (a|the) (.{1,20})( then| and|$|   get)",(None,3,1)],
    [r"(toward|towards|to) (the|a) (.{1,20})$",(None,3,None)],
    [r"(keep|keeping) (.{1,20}) (at|on|to) (your|the) (right|left)",(None,2,50)],

    # obj on your one side
    [r"(pass by|past|pass|reach|reach near|reaching|get to|pass around|loop around) the (.{1,20}) on your (right|left) side",(1,2,30)],
    [r"(pass by|past|pass|reach|reach near|reaching|get to|pass around|loop around) the (.{1,20}) with (.{1,20}) on your (right|left) side",(1,2,40)],
    [r"(pass by|past|pass|reach|reach near|reaching|get to|pass around|loop around) (.{1,20}) on your (right|left) side",(1,2,30)],
    [r"(pass by|past|pass|reach|reach near|reaching|get to|pass around|loop around) (.{1,20}) with (.{1,20}) on your (right|left) side",(1,2,40)],

    # clockwise circle around
    [r"(drive|move|go|head|heading|walk|fly) (behind|around|round) the (right|left) of the (.{1,20})$",(2,3,None)],
    [r"(go|circle) the (.{1,20}) on it  s (right|left) (side)?",(1,2,3)],
    [r"(go|loop|circle) (around|round) the (.{1,20}) (clockwise|counter clockwise|counterclockwise) (.{1,4}) (degrees|degree)",(456,3,None)],
    [r"(go|circle) the (.{1,20}) (clockwise|counter clockwise|counterclockwise) (.{1,4}) (degrees|degree)",(345,2,None)],
    [r"around the (.{1,20}) (clockwise|counter clockwise|counterclockwise) (.{1,4}) (degrees|degree)",(234,1,None)],
    [r"(go|loop|circle) (around|round) the (.{1,20}) (clockwise|counter clockwise|counterclockwise)",(24,3,None)],
    [r"(go|circle) the (.{1,20}) (clockwise|counter clockwise|counterclockwise)",(3,2,None)],
    [r"around the (.{1,20}) (clockwise|counter clockwise|counterclockwise)",(2,1,None)],
    [r"(go|loop|circle) (around|round) it (.{1,4}) (degrees|degree)",(234,"it",None)],
    [r"around the (.{1,20}) with it on your (right|left)",("around",1,20)],
    [r"(go|loop) (around|round) the (.{1,20}) (on|from|for) the (left|right) side",(None,2,4)],
    [r"(around|round) the (.{1,20}) (on|from|for) the (left|right) side",(1,2,4)],
    [r"rotate until you are facing the (.{1,20}) stop",("face",1,None)],
    [r"(go|loop) around (.{1,20}) from the (left|right) side",("around",1,2)],
    [r"hook around the (left|right) of the (.{1,20})($|\s\s)",("around",2,1)],
    [r"hook (left|right) around the (.{1,20})$",("around",2,1)],
    [r"circle around the (left|right) of the (.{1,20})$",("around",2,1)],
    [r"circle around the (left|right) of the (.{1,20}) (till|before|unril)",("around",2,1)],
    [r"circle around to the (right|left) of the (.{1,20})( and| then|$)",("around",2,1)],

    # just side of obj
    [r"(stop|stay) (on|at|to) the (right|left) hand side of (the|a) (.{1,20})$",(12,5,3)],
    [r"(stop|stay) (on|at|to) the (right|left) side of (the|a) (.{1,20})$",(12,5,3)],
    [r"(on|at|to) the (right|left) hand side of (the|a) (.{1,20})$",(None,4,2)],
    [r"(on|at|to) the (right|left) side of (the|a) (.{1,20})$",(None,4,2)],
    [r"stop (right|left) of (.{1,20})($|\s\s)",(None,2,1)],
    [r"(stop|stay) (on|at|to) the (right|left) of (the|a) (.{1,20})$",(None,5,3)],
    [r"(on|at|to) the (right|left) of (the|a) (.{1,20})$",(None,4,2)],
    [r"(stop|stay) (on|at|to) the (right|left) of (.{1,20})$",(None,4,3)],
    [r"(stop|stay) (before|in front of) (the|a) (.{1,20})$",(None,4,2)],
    [r"(on|at|to) the (right|left) of (.{1,20})$",(None,3,2)],

    [r"turn (.{1,4} degree) to your (left|right)", (1, None, 2)],
    [r"turn about (.{1,4} degree) to your (left|right)", (1, None, 2)],
    [r"turn (.{1,4} degrees) to your (left|right)", (1, None, 2)],
    [r"turn about (.{1,4} degrees) to your (left|right)", (1, None, 2)],
    [r"(stop|go|reach|reaching|reach near) (in front) of (.{1,20})$",(1,3,2)],
    [r"(stop|go|reach|reaching|reach near) (in front|right side|left side) of (the|a) (.{1,20})$",(1,4,2)],
    [r"(stop|go|reach|reaching|reach near) (in front|right side|left side) (.{1,20})$",(1,3,2)],

    [r"(curve left|curve right) (around|round) (.{1,20}) facing (.{1,20})",(12,3,None), ("face", 4, None)],
    [r"(curve left|curve right) (around|round) (.{1,20})$",(12,3,None)],
    [r"(curve|fly) (around|round) (.{1,20}) keeping (.{1,20}) (right|left)",(12,3,50)],
    [r"take a curve over (.{1,20}) (and|then)",("curve over",1,None)],
    [r"take a curve over (.{1,20})$",("curve over",1,None)],
    [r"(curve left|curve right) of (.{1,20})$",(1,2,None)],
    [r"curve the (left|right) of (.{1,20})$",("curve",2,1)],
    [r"curve the (left|right) side of (.{1,20})$",("curve",2,1)],

    # cross obj to another obj
    [r"cross the (.{1,20}) (to|to get to|to reach|reaching|reach near) (the|a) (.{1,20})$",(2,4,None)],

    # not so much matched instruction
    [r"take (right|left) near the (.{1,20})($|\s\s)",(1,2,None)],
    [r"go near the (.{1,20})$",(None,1,None)],
    [r"go straight down the (.{1,20})$",("go straight down",1,None)],
    [r"go straight down by the (.{1,20}) until you pass the (.{1,20})$",("go straight down",1,None),(None,2,None)],
    [r"stop when you (pass|past) (the front of|a|the) (.{1,20})",(1,3,None)],
    [r"(make|take) (a .{1} turn) (at|around|once you reach|behind) the (.{1,20})( and|$)",(12,4,None)],

    # just actions e.g. "then go straight" "and around it" "and loop around"

    # not match
    [r"circle (around) (it) leaving it on your (right|left)",("around","it",30)],

    # simple with start and end
    [r"^and loop around the (.{1,20})$",("around",1,None)],
    [r"^loop around the (.{1,20})$",("around",1,None)],
    [r"^and (loop around)$",(1,None,None)],
    [r"^around the (.{1,20})$",("around",1,None)],
    [r"^and around the (.{1,20})$",("around",1,None)],
    [r"^(go|walk) (straight)$",("straight",None,None)],
    [r"^(and|again|then) (go|walk) (straight)$",(3,None,None)],
    [r"^(take|turn) (right|left)$",(2,None,None)],
    [r"^(and|again|then) (take|turn) (right|left)$",(3,None,None)],
    [r"(pass|past|passing) the (.{1,20}) (in front)", (1,2,3)],
    [r"(pass|past|passing) the (.{1,20}) and", (1,2,None)],
    [r"(pass|past|passing) the (.{1,20})$", (1,2,None)],
    [r"take a (right|left) past (a|the) (.{1,20})", (1,3,None)],
    [r"turn around (.{1,20}) and",("around", 1, None)],
    [r"move (in front|left|right) of (.{1,20})",(None, 2, 1)],

    # last simple match
    [r"(cross|crossing) the (.{1,20}) on your (left|right)",(1,2,30)],
    [r"(cross|crossing) the (.{1,20}) (we made|while|and move|in front of)",(1,2,None)],
    [r"(cross|crossing) the (.{1,20})$",(1,2,None)],
    [r"(cross|crossing) (.{1,20})$",(1,2,None)],
    [r"(go|head|heading) (in front of the left|in front of the right) side of the (.{1,20})",(None,3,2)],
    [r"(left|right|back) side of (.{1,20})$",(None,2,1)],
    [r"(touch|touching) the (.{1,20})$",(1,2,None)],
    [r"keep moving forward (towards) (.{1,20})$",(1,2,None)],
    [r"keep moving forward (towards) (.{1,20}) near",(1,2,None)],
    [r"(face) (the|a) (.{1,20})$",(1,2,None)],
    [r"(turn|take a|make a) (straight|left|right)",(2,None,None)],
    [r"in front of the (.{1,20})$",(None,1,"in front")],
    [r"turn go straight$", ("straight", None, None)],
    [r"then take a sharp left$", ("sharp left", None, None)],
    [r"^when reached$",(None,"tree",None)],

]

# test sentence
if __name__ == '__main__':
    instr = "walk around the stone on the side facing the well"
    if instr == "-1":
        exit(-1)
    for rule in RULES:
        match = re.search(rule[0], instr)
        if match:
            print(rule[0], "-->", instr[match.regs[0][0]:match.regs[0][1]])
