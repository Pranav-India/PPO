
import pyodigo
from tf_ppo import PPO

def main(mode):
    pyodigo.StartSimulation()
    pyodigo.SetMaxSteeringAngle(30)
    pyodigo.SetMaxMotorTorque(0.5)
    if type(mode) != int and len(mode) != 1:
        print("please Select correct mode among 1 or 2")
    if mode == 1:
        agent = PPO()
        agent.learn(total_timesteps=200_000_000)


if __name__ == '__main__':
    mode = int(input("Enter mode 1.Training 2.Testing"))
    main(mode)
