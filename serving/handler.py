import yaml


def load_conf_file(config_file):
   with open(config_file, "r") as f:
       config = yaml.safe_load(f)
       return config


def create_WLASL_dictionary(wlasl_class_list_file):
    
    global wlasl_dict 
    wlasl_dict = {}
    
    with open(wlasl_class_list_file) as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value
    return wlasl_dict

