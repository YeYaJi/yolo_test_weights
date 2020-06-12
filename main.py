import os
import yolo_test_batch_for_dibang

weights_path = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/batch_weights"
save_path = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/result_txt"
txt_correct_path = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/correct_txt"
img_path = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/test_img"
cfg_path = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/yolov3-tiny.cfg"
voc_data = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/voc.data"

weights_name_list = os.listdir(weights_path)


def result_txt_creat(cfg_path, weights_name_list, voc_data, save_path):
    for i, weight in enumerate(weights_name_list):
        save_path_filename = os.path.join(save_path, weight)
        os.mkdir(save_path_filename[:-8])
        weight_path = os.path.join(weights_path, weight)
        areyouok = yolo_test_batch_for_dibang.DETECT_YOLO()
        areyouok.creat_txt(cfg_path=cfg_path, weight_path=weight_path, voc_data=voc_data, img_path=img_path,
                           save_path=save_path_filename[:-8])


# result_txt_creat(cfg_path, weights_name_list, voc_data, save_path)


def contrast_txt(txt_correct_path, txt_test_path):
    txt_correct_name_list = os.listdir(txt_correct_path)
    txt_test_name_list = os.listdir(txt_test_path)
    wrong_num = 0
    for i in range(len(txt_correct_name_list)):  # 一个txt
        if txt_correct_name_list[i] not in txt_test_name_list:
            wrong_num += 1
        else:
            correct_list = []
            test_list = []

            with open(os.path.join(txt_correct_path, txt_correct_name_list[i])) as cf:
                correct_read_list = cf.readlines()
                for j in range(len(correct_read_list)):  # txt中的一行
                    print("correct_read_list[j][0]=", correct_read_list[j][0])
                    correct_list.append(int(correct_read_list[j][0]))
                correct_list.sort()

            with open(os.path.join(txt_test_path, txt_correct_name_list[i])) as tf:
                test_read_list = tf.readlines()
                for j in range(len(test_read_list)):
                    print("test_read_list[j][0]==", test_read_list[j][0])
                    test_list.append(int(test_read_list[j][0]))
                test_list.sort()
            if correct_list != test_list:
                wrong_num += 1
    print("wrong_num=%d" % wrong_num)
    return wrong_num

for i in
contrast_txt(txt_correct_path, txt_test_path)
