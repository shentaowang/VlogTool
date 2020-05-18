from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import cv2
import torch
import pickle
import numpy as np
from scipy.spatial.distance import cdist
import shutil

def face_extract(data_path, face_size):
    """
    :param data_path: the file for image to detect face and extract the embedding
    :param face_size: the extract face size
    :return: for each image save the face and the embedding
    """
    save_path = data_path+"_extract"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_paths = os.listdir(data_path)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=face_size, post_process=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for image_path in image_paths:
        out_path = os.path.join(save_path, image_path.split('.')[0])
        if(not os.path.exists(out_path)):
            os.makedirs(out_path)
        image = cv2.imread(os.path.join(data_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(image)
        if boxes is not None:
            for idx, box in enumerate(boxes):
                prob=probs[idx]
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]
                face = cv2.resize(face, (face_size, face_size))
                input = torch.from_numpy((face-127.5)/128)
                input = input.unsqueeze(0).permute(0, 3, 1, 2).to(device, dtype=torch.float)
                embedding = resnet(input).detach().cpu()
                face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_path, "face-{:0>4d}-{:.4f}.jpg".format(idx, prob)), face)
                fout = open(os.path.join(out_path, "embedding-{:0>4d}-{:.4f}.pkl".format(idx, prob)), 'wb')
                pickle.dump(embedding.numpy()[0], fout)
                fout.close()

    return

def face_cluster(data_file, thr_same=0.4, thr_face=0.9):
    """
    input the embeddings, out two dictionary. one for cluster the face part image,  one  for cluster the original part
    image, cation only support .jpg and .png
    :param data_file: the dictionary contain the full image. which include the face and embedding
    :param thr_same: the threshold for one image
    :param thr_face: the threshold for face
    :return:
    """
    save_face_path = data_file + '_face'
    save_orig_path = data_file + '_orig'
    if not os.path.exists(save_face_path):
        os.makedirs(save_face_path)
    if not os.path.exists(save_orig_path):
        os.makedirs(save_orig_path)
    embedding_face_file = data_file+'_extract'
    embeddings = []
    image_files = []
    face_files = []
    for image_file in os.listdir(embedding_face_file):
        tmp = []
        for face_file in os.listdir(os.path.join(embedding_face_file, image_file)):
            # filter the low confidence faces
            if '.jpg' in face_file and float(face_file.split('-')[-1][0:6])>thr_face:
                face_files.append(os.path.join(embedding_face_file, image_file, face_file))
                embeding_file = face_file[:-4] + '.pkl'
                embeding_file = embeding_file.replace('face', 'embedding')
                with open(os.path.join(embedding_face_file, image_file, embeding_file), 'rb') as fin:
                    embed = pickle.load(fin)
                    # embed = embed.numpy()
                    # tmp.append(embed[0])
                    tmp.append(embed)
        if len(tmp)>0:
            embeddings += tmp
            image_files += [os.path.join(data_file, image_file)]*len(tmp)

    embeddings = np.array(embeddings)
    image_files = np.array(image_files)
    face_files  = np.array(face_files)
    dist = cdist(embeddings, embeddings, 'euclidean')
    cnt_id = 0
    # a simple greedy strategy
    visited = np.zeros((embeddings.shape[0]))
    for i in range(dist.shape[0]):
        part = dist[i]
        mask = ~((part>thr_same) + (visited==1))
        group_embed = embeddings[mask]
        if(group_embed.shape[0]>1):
            face_identity = os.path.join(save_face_path, str(cnt_id))
            if not os.path.exists(face_identity):
                os.makedirs(face_identity)
            orig_identity = os.path.join(save_orig_path, str(cnt_id))
            if not os.path.exists(orig_identity):
                os.makedirs(orig_identity)
            with open(os.path.join(face_identity, 'embedding.pkl'), 'wb') as fout:
                pickle.dump(group_embed, fout)
            visited[mask] = 1


            for idx, path in enumerate(face_files[mask]):
                shutil.copy(path, os.path.join(face_identity, str(idx) + '-' + path.split(os.sep)[-1]))

            cnt_id+=1
            group_image = set(image_files[mask].tolist())
            with open(os.path.join(face_identity, 'images.txt'), 'w') as fout:
                for image_file in group_image:
                    fout.write(image_file+'\n')

            for path in group_image:
                image_copied = 0
                if os.path.exists(path+'.jpg'):
                    shutil.copy(path+'.jpg', os.path.join(orig_identity, path.split(os.sep)[-1])+'.jpg')
                    image_copied = 1
                if os.path.exists(path+'.png'):
                    shutil.copy(path+'.png', os.path.join(orig_identity, path.split(os.sep)[-1])+'.png')
                    image_copied = 1
                assert image_copied == 1


if __name__ == "__main__":
    data_path = '../data/data20200402'
    face_size = 160
    # face_extract(data_path, face_size)
    face_cluster(data_path, 0.8, 0.9)

