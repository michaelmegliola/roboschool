import gym, roboschool
import numpy as np
import math

lower_joint = [-math.pi/2.0,-math.pi/3.0]
upper_joint = [-math.pi/4.0, math.pi/4.0]
trans = np.array((1,-1,1,1,1,1,1,-1))

def get_state_n(state):
    state_n = 0
    for n in range(len(state)):
        state_n += state[n] * 2**n
    return state_n

def get_state(n):
    state = np.zeros((8))
    for i in range(0,8):
        state[i] = 1 if n % 2 > 0 else 0
        n //= 2
    return state

def get_action(state):
    action = np.zeros((8))
    for n in range(4):
        action[n*2] = upper_joint[int(state[n*2])]
        action[n*2+1] = lower_joint[int(state[n*2+1])]
    return np.multiply(action, trans)

def sample(v):
    b = np.random.randint(8)
    s = np.copy(v)
    s[b] = 1 if v[b] == 0 else 0
    return s

def demo_run():
    env = gym.make("JvonAnt-v0")

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        q = np.zeros((256,8))

        state = [0,0,0,0,0,0,0,0]

        explore = 0.10

        while 1:
            av = sample(state)
            obs, r, done, _ = env.step(get_action(av))
            print(state,av)
            state = av

            #obs, r, done, _ = env.step(np.array([upper_joint[i],lower_joint[i],upper_joint[i],lower_joint[i],upper_joint[i],lower_joint[i],upper_joint[i],lower_joint[i]]))

            score += r
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                restart_delay = 60*2  # 2 sec at 60 fps
            else:
                restart_delay -= 1
                if restart_delay==0: break

if __name__=="__main__":
    demo_run()
