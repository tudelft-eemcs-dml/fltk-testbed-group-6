import subprocess
import yaml
from fltk.util.base_config import BareConfig
import os


def run_all(name='fl', num=5, path='./results'):
    """
        run experiment according to the configuration of experiment.yaml

    :param out_name:
    :param err_name:
    :param num:
    :param path:
    :return:
    """
    err_path = os.path.join(path, name+'-err.txt')
    out_path = os.path.join(path, name+'-out.txt')
    with open(out_path, "wb+") as out, open(err_path, "wb+") as err:
        subprocess.Popen("python3 -m fltk single configs/experiment.yaml --rank=0", shell=True, stdout=out, stderr=err)
        for i in range(num):
            subprocess.Popen("python3 -m fltk single configs/experiment.yaml --rank={}".format(str(i+1)), shell=True, stdout=out, stderr=err)


if __name__ == '__main__':
    config_path = './configs/experiment.yaml'
    with open(config_path) as file:
        cfg = BareConfig()
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        cfg.merge_yaml(yaml_data)

    # if cfg.improve == 0:
    #     name = f'{cfg.attack_type}-{cfg.aggregation_rule}'
    # else:
    #     name = f'{cfg.attack_type}-{cfg.aggregation_rule}-{cfg.improve}-{cfg.improve_data_ratio}'
    name = 'fl'
    run_all(name=name, num=cfg.world_size-1)
