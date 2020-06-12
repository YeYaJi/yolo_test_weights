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
    """

    :param cfg_path:
    :param weights_name_list:
    :param voc_data:
    :param save_path:
    :return:
    """
    for i, weight in enumerate(weights_name_list):
        save_path_filename = os.path.join(save_path, weight)
        os.mkdir(save_path_filename[:-8])
        weight_path = os.path.join(weights_path, weight)
        yolo_test_batch_for_dibang.creat_txt(cfg_path=cfg_path, weight_path=weight_path, voc_data=voc_data,
                                             img_path=img_path,
                                             save_path=save_path_filename[:-8])


# result_txt_creat(cfg_path, weights_name_list, voc_data, save_path)


def contrast_txt(txt_correct_path, txt_test_path):
    """

    :param txt_correct_path:
    :param txt_test_path:
    :return:
    """
    txt_correct_name_list = os.listdir(txt_correct_path)
    txt_test_name_list = os.listdir(txt_test_path)
    wrong_num = 0
    for i in range(len(txt_correct_name_list)):  # 一个txt
        if txt_correct_name_list[i] not in txt_test_name_list:
            wrong_num += 1
            print("没检测到", txt_correct_name_list[i])
        else:
            correct_list = []
            test_list = []

            with open(os.path.join(txt_correct_path, txt_correct_name_list[i])) as cf:
                correct_read_list = cf.readlines()
                for j in range(len(correct_read_list)):  # txt中的一行
                    # print("correct_read_list[j][0]=", correct_read_list[j][0])
                    correct_list.append(int(correct_read_list[j][0]))
                correct_list.sort()

            with open(os.path.join(txt_test_path, txt_correct_name_list[i])) as tf:
                test_read_list = tf.readlines()
                for j in range(len(test_read_list)):
                    # print("test_read_list[j][0]==", test_read_list[j][0])
                    test_list.append(int(test_read_list[j][0]))
                test_list.sort()
            if correct_list != test_list:
                wrong_num += 1
                print('correct==', correct_list)
                print('test==', test_list,"\n")

    print("wrong_num=%d" % wrong_num)

    print("正确文件个数%d" % len(txt_correct_name_list))
    print("测试文件个数%d\n" % len(txt_test_name_list))

    return wrong_num


if __name__ == "__main__":
    txt_file_test_list = os.listdir(save_path)

    for txt_file_name in txt_file_test_list:
        txt_path = os.path.join(save_path, txt_file_name)
        contrast_txt(txt_correct_path, txt_path)
