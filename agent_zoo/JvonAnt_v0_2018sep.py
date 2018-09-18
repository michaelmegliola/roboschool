import gym, roboschool
import numpy as np

def demo_run():
    env = gym.make("JvonAnt-v0")

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        while 1:
            obs, r, done, _ = env.step(env.action_space.sample())
            print(obs)
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
