import os
import gc

# with open("/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/2.txt","a") as f:
#     f.write("qqq")
# a=[1,3,66,2]
# a.sort()
# print(a)
txt_correct_path = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/correct_txt"
txt_test_path = "/home/gsh/PycharmProjects/PycharmProjects/PycharmProjects/yolo_net/yolo_text_weights/result_txt/yolov3-tiny_20000"


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


contrast_txt(txt_correct_path, txt_test_path)
