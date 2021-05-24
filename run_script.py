import subprocess
import yaml
from fltk.util.base_config import BareConfig


def run_all(compromised=2, num=5):
    """
    run experiment according to the configuration of experiment.yaml
    :param curr: 
    :param compromised: 
    :param num: 
    :return: 
    """
    with open("flavg_stdout.txt", "wb") as out, open("flavg_stderr.txt", "wb") as err:
        subprocess.Popen("python3 -m fltk single configs/experiment.yaml --rank=0", shell=True, stdout=out,stderr=err)
        for i in range(num):
            subprocess.Popen("python3 -m fltk single configs/experiment.yaml --rank={}".format(str(i+1)), shell=True, stdout=out,stderr=err)


if __name__ == '__main__':
    config_path = './configs/experiment.yaml'
    with open(config_path) as file:
        cfg = BareConfig()
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        cfg.merge_yaml(yaml_data)
    print(cfg)

    run_all(cfg.compromised_num, cfg.world_size-1)
