import argparse
import yaml

import numpy as np

from pylabel import importer
from glob import glob
from os import mkdir, listdir, remove
from shutil import copy2, move, rmtree
from os.path import abspath, exists, join



PARAMS = []
DIRS = []


def parser():
    global PARAMS
    global DIRS

    p = argparse.ArgumentParser(prog="Dataset Splitter", description="Este programa separa um dataset yolo de acordo com o tamanho da bounding box em relação à imagem. Útil para dividir dataset em imagens mais fáceis ou difíceis de serem aprendidas pelo modelo")

    p.add_argument('dataset', help="Caminho para o dataset (caminho absoluto)")
    p.add_argument('data_name', default="data.yaml", help="Nome do arquivo .yaml com dados do dataset. (Default = data.yaml)", )

    p.add_argument('-p', '--percent', required=True, help="Percentual que a bounding-box deve preencher da imagem para ser colocada no novo dataset.", type=float)
    
    p.add_argument('-d', '--destination', help="Caminho do dataset de destino. Valor default é 'novo-dataset'", default="novo-dataset")

    p.add_argument('-s', '--split', action="store_true", help="True | False (Default True) Dita se o novo dataset deve ser partido em train, test, val.", default=True)

    p.add_argument('-i', '--invert', action="store_true", help="True | False (Default False) Alguns datasets vem com a partição invertida (na raiz, duas pastas: images e labels, com subpastas train, test e val) Esta flag faz com que o script leve isso em conta para fazer a avaliação.", default=False)

    PARAMS = p.parse_args()
    PARAMS.percent = PARAMS.percent / 100

    data = join(PARAMS.dataset, PARAMS.data_name)

    with open(data, 'r') as y:
        file = yaml.load(y, yaml.FullLoader)

        for i in ['train', 'test', 'val']:
            if PARAMS.invert:
                d = file[i].split('/')[-1]
                DIRS.append(d)
            else:
                d = file[i].split('/')[-2]
                DIRS.append(d)


def generate_statistics(labels: list, classes: int) -> list[int, list]:
    """
    A tupla retornada possui (<total de labels>, <lista, onde posição é o id da classe>)
    """
    count_total = 0
    count_cls = []

    for _ in range(classes):
        count_cls.append(0)

    for label in labels:
        label = label.strip("\n").strip().split(" ")
        class_id = int(label[0])

        count_cls[class_id] += 1
        count_total += 1

    return (count_total, count_cls)


def print_stats(info_cls: np.ndarray, names: list, total: int) -> None:
    for idx, i in enumerate(info_cls):
        t = (i / total) * 100
        print(f"Classe {names[idx]}: {t:.2f}%")


def calculate_bb_size(labels: list) -> list:
    r_sizes = []

    for label in labels:

        label = label.strip("\n").split(" ")

        if label[-1] != '':
            w, h = float(label[-2]), float(label[-1])

            r_sizes.append(w * h)

    return r_sizes


def calculate_percent(img_p: str, labels: list) -> bool:
    p = calculate_bb_size(labels)
    threshold = PARAMS.percent

    for i in p:
        if i >= threshold: return True
    
    return False


def find_corr_labels(label_dir: str, file: str) -> list:
    path = join(label_dir, file) + ".txt"
    labels = []

    if exists(path):
        with open(path, 'r') as f:
            labels = f.readlines()
        
        f.close()
    
    return labels


def save_new_dataset(data: list, dest: str) -> None:
    images = join(dest, 'images')
    labels = join(dest, "labels")

    y = join(PARAMS.dataset, PARAMS.data_name)

    if not exists(images):
        mkdir(images)

    if not exists(labels):
        mkdir(labels)

    copy2(y, dest)

    for img, label in data:
        copy2(img, images)
        copy2(label, labels)

    if PARAMS.split:
        dataset = importer.ImportYoloV5(path=labels, path_to_images=images)
        dataset.splitter.StratifiedGroupShuffleSplit(train_pct=0.6, test_pct=0.2, val_pct=0.2)

        dataset.export.ExportToYoloV5(output_path=PARAMS.destination,use_splits=True)
        
        remove("dataset.yaml")
        
        for i in DIRS:
            dest = abspath(PARAMS.destination)  # Aqui vai ser './novo-dataset'

            currdir = join(dest, i)  # Então, currdir = ./novo-dataset/<train, test ou val>

            l_dir = join(currdir, 'labels') 
            i_dir = join(currdir, 'images')

            if not exists(l_dir):
                mkdir(l_dir)
            
            if not exists(i_dir):
                mkdir(i_dir)
            
            for label in listdir(currdir):  # A label aqui é, por exemplo, ./novo-dataset/train/nome.txt
                eq_jpg = label[:-4] + ".*"
                
                i_path = join(images, eq_jpg)
                l_path = join(currdir, label)

                if exists(l_path) and glob(i_path):
                    
                    i_path = glob(i_path)[0]

                    print(f"[+] Saving {i_path}...")
                    move(i_path, i_dir)
                    print(f"[+] Saving {l_path}...")
                    move(l_path, l_dir)
        
        rmtree(images)
        rmtree(labels)
        
        print("[+] Atualizando data.yaml...")
        
        dest = abspath(PARAMS.destination)
        arq = join(dest, PARAMS.data_name)
    
        with open(arq, mode='r') as y:
            file = yaml.load(y, Loader=yaml.FullLoader)
            file['train'] = join(dest, 'train/images')
            file['test'] = join(dest, 'test/images')
            file['val'] = join(dest, 'val/images')
        
            y.close()
            
        with open(arq, 'w') as y:
            yaml.dump(file, y)
        
        print("[+] data.yaml Atualizado!")
            

def process_dataset():
    abs = PARAMS.dataset
    nc = 0
    classes_names = []

    if not exists(PARAMS.destination):
        mkdir(PARAMS.destination)

    dest_path = abspath(PARAMS.destination)
    data = join(PARAMS.dataset, PARAMS.data_name)

    with open(data, "r") as f:
        file = yaml.load(f, Loader=yaml.FullLoader)
        classes_names = file['names']
        nc = len(file['names'])
        
        f.close()

    total_annotations = 0
    ann_per_cls = np.ndarray(nc)

    if exists(abs):
        to_copy = []

        for p in DIRS:
            print(f"[+] Processando {p}")
            count = 0

            if PARAMS.invert:
                img_path = join(abs, "images")
                img_path = join(img_path, p)

                labels_path = join(abs, 'labels')
                labels_path = join(labels_path, p)

            else:
                img_path = join(abs, p)
                img_path = join(img_path, "images")

                labels_path = join(abs, p)
                labels_path = join(labels_path, 'labels')

            for img in listdir(img_path):
                name = img[:-4]
                labels = find_corr_labels(labels_path, name)

                count += 1

                if calculate_percent(join(img_path, img), labels):
                    stats = generate_statistics(labels, nc)
                    total_annotations += stats[0]
                    ann_per_cls = ann_per_cls + np.array(stats[1])

                    label_name = join(labels_path, name)
                    label_name += ".txt"

                    to_copy.append((join(img_path, img), label_name))

            print(f"{count} imagens processadas em {p}.")


        print(f'\n\n| RESULTADOS DA PARTIÇÃO |\n{len(to_copy)} imagens com bounding box >= {PARAMS.percent * 100:.2f}%. \n\nTotal de classes: {nc}')
        print_stats(ann_per_cls, classes_names,total_annotations)

        i = input(f"Continuar com a operação? [y/n] ")

        if i not in ["no", 'n']:
            save_new_dataset(to_copy, dest_path)

        else: 
            print("Operação abortada!")


if __name__ == '__main__':
    parser()
    process_dataset()