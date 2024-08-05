import cv2
import serial
import numpy as np
import math
import time
import copy
import serial
#import jetson.gpio
com_HMI = serial.Serial('COM27',115200,timeout = 0.2)
com = serial.Serial('COM25', 115200,timeout = 0.2)
angle_rad = 0
angle = 0
#angle_rad,abs_1_x,abs_2_x,abs_3_x,abs_4_x,abs_5_x,abs_6_x,abs_7_x,abs_8_x,abs_9_x,abs_1_y,abs_2_y,abs_3_y,abs_4_y,abs_5_y,abs_6_y,abs_7_y,abs_8_y,abs_9_y,abs_b_1_x,abs_b_2_x,abs_b_3_x,abs_b_4_x,abs_b_5_x,abs_b_1_y,abs_b_2_y,abs_b_3_y,abs_b_4_y,abs_b_5_y,abs_w_1_x,abs_w_2_x,abs_w_3_x,abs_w_4_x,abs_w_5_x,abs_w_1_y,abs_w_2_y,abs_w_3_y,abs_w_4_y,abs_w_5_y = 0
abs_1_x = 76
abs_1_y = 56
abs_2_x = 106
abs_2_y = 56
abs_3_x = 136
abs_3_y = 56

abs_4_x = 76
abs_4_y = 88
abs_5_x = 106
abs_5_y = 88
abs_6_x = 138
abs_6_y = 88

abs_7_x = 76
abs_7_y = 120
abs_8_x = 107
abs_8_y = 120
abs_9_x = 139
abs_9_y = 120

Positions = [[75,55],[106,55],[141,55],[77,86],[108,86],[141,86],[77,120],[108,120],[143,12],
             [38,26],[38,55],[38,92],[38,125],[38,152],
             [180,26],[180,56],[180,87],[180,120],[180,152]]




state_1 = 0
state_2 = 0
state_3 = 0
state_4 = 0
state_5 = 0
state_6 = 0
state_7 = 0
state_8 = 0
state_9 = 0
width = 0
height = 0
black_1_state = 0
black_2_state = 0
black_3_state = 0
black_4_state = 0
black_5_state = 0
white_1_state = 0
white_2_state = 0
white_3_state = 0
white_4_state = 0
white_5_state = 0
k_real_frame = 0
x_plus = -5.43
y_plus = -0.9
# 0-->空  1 -->白   2-->黑
board = [[0 for _ in range(3)] for _ in range(3)]
threshold = 400                #棋子大小阈值 
white_value_low = 160   #白色识别亮度，一般在150左右，最大值固定255
white_saturation_high = 70   #白色识别饱和度，一般在60左右，最小值固定为0
threshold_cheese = 80  #棋盘轮廓 0-255
blue_saturation_low = 50  #绿色饱和度一般在100以下
blue_Hue_low = 30    #绿色色调低
blue_Hue_high = 90      #绿色色调高
black_value_high = 90
blue_value_low = 50

num_matrices = 3
#board_steps = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(num_matrices)]
board_steps = [[[0,0,0],[0,0,0],[0,0,0]]]
board_black = [0 for _ in range (5)]
board_white = [0 for _ in range (5)]
#初始化摄像头
cap = cv2.VideoCapture('http://192.168.2.41:4747/video?640x480')

# 设置图像处理参数
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
cap.set(cv2.CAP_PROP_FPS, 30)



def arr_to_num(arr_position):
    return (arr_position[0] * 3) + arr_position[1] + 1


def check_cheat(color):
    global board_steps
    cur_board = board_steps[-1]
    pre_board = board_steps[-2]
    cheat_flag = 0
    position = [[0, 0], [0, 0]]
    for i in range(3):
        for j in range(3):
            if pre_board[i][j] == color and cur_board[i][j] != color:
                cheat_flag = 1
                position[0] = [i, j]
                print("User Cheat!")

    if cheat_flag == 0:
        return [[-1, -1], [-1, -1]]

    for i in range(3):
        for j in range(3):
            if pre_board[i][j] == 0 and cur_board[i][j] != 0:
                position[1] = [i, j]

    board_steps.pop()

    return position


def correct_cheat(position):
    from_p2p(arr_to_num(position[0]), arr_to_num(position[1]))


def parameter_adjustment(data):
    global threshold_cheese,white_saturation_high,white_value_low,black_value_high,blue_Hue_low,blue_Hue_high,blue_saturation_low,blue_value_low
          
    if data[1]== 0x01:
        threshold_cheese = int(data[2])#反二值化参数，调节该bar改变棋盘识别
    elif data[1] == 0x02:
        white_saturation_high = int(data[2])#白色识别亮度，一般在150左右，最大值固定255
    elif data[1] == 0x03:
        white_value_low  = int(data[2])#白色识别饱和度，一般在60左右，最小值固定为0
    elif data[1] == 0x04:
        black_value_high = int(data[2])#黑棋V上限调节
    elif data[1] == 0x05:
        blue_Hue_low = int(data[2])#绿棋H下限调节
    elif data[1] == 0x06:
        blue_Hue_high = int(data[2])#绿棋H上限调节
    elif data[1] == 0x07:
        blue_saturation_low = int(data[2])#绿棋S下限调节
    elif data[1] == 0x08:
        blue_value_low = int(data[2])#绿棋V下限调节

def print_board(board):
    # for row in board:
    #     print(" | ".join(row))
    #     print("-" * 5)
    print(board)


def check_winner(board, player):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]],
    ]
    return [player, player, player] in win_conditions


def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]


def player_move(board):
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if board[move // 3][move % 3] == 0:
                board[move // 3][move % 3] = 2
                break
            else:
                print("Cell already taken, try again.")
        except (ValueError, IndexError):
            print("Invalid input, enter a number between 1 and 9.")


def minimax(board, depth, is_maximizing):
    if check_winner(board, 1):
        return 1
    if check_winner(board, 2):
        return -1
    if not get_empty_cells(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for (i, j) in get_empty_cells(board):
            board[i][j] = 1
            score = minimax(board, depth + 1, False)
            board[i][j] = 0
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for (i, j) in get_empty_cells(board):
            board[i][j] = 2
            score = minimax(board, depth + 1, True)
            board[i][j] = 0
            best_score = min(score, best_score)
        return best_score

def minimax_cf(board, depth, is_maximizing):
    if check_winner(board, 2):
        return 1
    if check_winner(board, 1):
        return -1
    if not get_empty_cells(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for (i, j) in get_empty_cells(board):
            board[i][j] = 2
            score = minimax_cf(board, depth + 1, False)
            board[i][j] = 0
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for (i, j) in get_empty_cells(board):
            board[i][j] = 1
            score = minimax_cf(board, depth + 1, True)
            board[i][j] = 0
            best_score = min(score, best_score)
        return best_score

index = 0
def computer_move(board):
    global index
    best_score = -math.inf
    best_move = None
    for (i, j) in get_empty_cells(board):
        board[i][j] = 1
        score = minimax(board, 0, False)
        board[i][j] = 0
        if score > best_score:
            best_score = score
            best_move = (i, j)
    board[best_move[0]][best_move[1]] = 1
    board_steps.append(board)
    print(best_move)
    print(board)
    from_p2p(index + 15, arr_to_num([best_move[0],best_move[1]]))
    time.sleep(1)
    from_p2p(index + 15, arr_to_num([best_move[0],best_move[1]]))
    time.sleep(23)######################################################记得改大一点
    index += 1 
    computer_move_to_position = Positions[arr_to_num([best_move[0],best_move[1]])-1]
    print(computer_move_to_position)
    return computer_move_to_position
    # time.sleep(10)
    # cmd_go(0,0)
    # while True:
    #     if com.in_waiting > 0 :
    #         data_arm = com.read(1)#从机械臂收回放置完成指令
    #         received_byte = int.from_bytes(data_arm, byteorder='little')
    #         if received_byte == 0x99:
    #             print("GO 0 Done!")
    #             break


def is_full(board):
    for row in board:
        if 0 in row:
            return False
    return True


def check_winner_cf(board):
    # 检查行、列和对角线
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0

def minimax_cf(board, depth, is_maximizing):
    winner = check_winner_cf(board)
    if winner != 0:
        return (10 if winner == 2 else -10) - depth
    if is_full(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = 2
                    score = minimax_cf(board, depth + 1, False)
                    board[i][j] = 0
                    best_score = max(best_score, score)
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = 1
                    score = minimax_cf(board, depth + 1, True)
                    board[i][j] = 0
                    best_score = min(best_score, score)
        return best_score






index_cf = 1
def computer_move_cf(board):
    global index_cf
    best_score = -math.inf
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = 2
                score = minimax_cf(board, 0, False)
                board[i][j] = 0
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    board[best_move[0]][best_move[1]] = 2
    board_steps.append(board)
    print(best_move)
    print(board)
    from_p2p(index_cf + 10, arr_to_num([best_move[0],best_move[1]]))
    time.sleep(1)
    from_p2p(index_cf + 10, arr_to_num([best_move[0],best_move[1]]))
    time.sleep(23)###########################################################记得改大一点
    index_cf += 1 
    computer_move_to_position = Positions[arr_to_num([best_move[0],best_move[1]])-1]
    print(computer_move_to_position)
    return computer_move_to_position

def req_board():
    
    global angle_rad,frame,mask_black,mask_blue,mask_white,white_1_state,white_2_state,white_3_state,white_4_state,white_5_state,black_1_state,black_2_state,black_3_state,black_4_state,black_5_state,state_1,state_2,state_3,state_4,state_5,state_6,state_7,state_8,state_9,threshold_cheese,white_saturation_high,white_value_low,black_value_high,blue_Hue_low,blue_Hue_high,blue_saturation_low,blue_value_low
    ret, frame = cap.read()
    window_handle = cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE) 
    
    lower_black = np.array([0, 0, 0])       #设置黑色识别阈值
    upper_black = np.array([190, 255, black_value_high])       #设置黑色识别阈值

    lower_blue = np.array([blue_Hue_low, blue_saturation_low, blue_value_low])       #设置蓝色识别阈值
    upper_blue = np.array([blue_Hue_high, 255, 255])       #设置蓝色识别阈值

    lower_white = np.array([0, 0, white_value_low])       #设置白色识别阈值
    upper_white = np.array([180, white_saturation_high, 255])  #设置白色识别阈值
    global hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#色彩空间转换
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)  #取出蓝色空间
    mask_blue = cv2.GaussianBlur(mask_blue,(5,5),0,0) #高斯滤波
    mask_white = cv2.inRange(hsv,lower_white,upper_white)#取出白色空间
    mask_white = cv2.GaussianBlur(mask_white,(5,5),0,0)#高斯滤波
    mask_black = cv2.inRange(hsv,lower_black,upper_black)#取出hei色空间
    mask_black = cv2.GaussianBlur(mask_black,(5,5),0,0)#高斯滤波
    
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#找棋盘外轮廓
    flag = 0
    
    global board_black
    global board_white
    
    for contour in contours:
        if cv2.arcLength(contour,True)>700 and cv2.arcLength(contour,True)<1300:#######################################
            flag = 1
            rect = cv2.minAreaRect(contour)    #矩形拟合棋盘外廓
            points = cv2.boxPoints(rect)               #转换为drawcounters可用
            points = np.int0(points)
            cv2.drawContours(frame,[points],0,(0,0,255),2)#画出棋盘轮廓
            center_x ,center_y = rect[0] #棋盘中心点坐标
            width , height= rect[1] #棋盘的长款，宽度
            angle = rect[2] #棋盘旋转角度
            angle_rad = math.radians(angle)#转换为弧度制
            left_x = center_x + (height/2)*(math.cos(angle_rad)) - (width/2)*(math.sin(angle_rad))#棋盘左下角坐标
            left_y = center_y + (height/2)*(math.cos(angle_rad)) + (width/2)*(math.sin(angle_rad))

            #求出棋盘上九个格子的中心点位置
            global cheese_1_x,cheese_1_y,cheese_2_x,cheese_2_y,cheese_3_x,cheese_3_y,cheese_4_x,cheese_4_y,cheese_5_x,cheese_5_y,cheese_6_x,cheese_6_y,cheese_7_x,cheese_7_y,cheese_8_x,cheese_8_y,cheese_9_x,cheese_9_y

            #棋子1
            cheese_1_x = center_x - (width/3)*(math.cos(angle_rad)) - (height/3)*(math.sin(angle_rad))#一号棋子x坐标
            cheese_1_y = center_y - (width/3)*(math.sin(angle_rad)) + (height/3)*(math.cos(angle_rad))#一号棋子y坐标
            cheese_1_x_l ,cheese_1_x_r = int (cheese_1_x)-22 ,int(cheese_1_x+22) #roi用
            cheese_1_y_l ,cheese_1_y_r= int (cheese_1_y)-22 ,int(cheese_1_y)+22
            roi_1_b = mask_black[cheese_1_y_l:cheese_1_y_r,cheese_1_x_l:cheese_1_x_r] #反二值化
            roi_1_blue = mask_blue[cheese_1_y_l:cheese_1_y_r,cheese_1_x_l:cheese_1_x_r]
            roi_1_white = mask_white[cheese_1_y_l:cheese_1_y_r,cheese_1_x_l:cheese_1_x_r]
            if cv2.countNonZero(roi_1_blue)>threshold:
                state_1 = 0
            if cv2.countNonZero(roi_1_white)>threshold:
                state_1 = 1
            if cv2.countNonZero(roi_1_b)>threshold:
                state_1 = 2
    
            
            #棋子2
            cheese_2_x = center_x - (width/3)*(math.sin(angle_rad))
            cheese_2_y = center_y +(height/3)*(math.cos(angle_rad))
            cheese_2_x_l ,cheese_2_x_r = int (cheese_2_x)-22 ,int(cheese_2_x+22) #roi用
            cheese_2_y_l ,cheese_2_y_r= int (cheese_2_y)-22 ,int(cheese_2_y)+22
            roi_2_b = mask_black[cheese_2_y_l:cheese_2_y_r,cheese_2_x_l:cheese_2_x_r] #反二值化
            roi_2_blue = mask_blue[cheese_2_y_l:cheese_2_y_r,cheese_2_x_l:cheese_2_x_r]
            roi_2_white = mask_white[cheese_2_y_l:cheese_2_y_r,cheese_2_x_l:cheese_2_x_r]
            if cv2.countNonZero(roi_2_blue)>threshold:
                state_2 = 0
            if cv2.countNonZero(roi_2_white)>threshold:
                state_2 = 1
            if cv2.countNonZero(roi_2_b)>threshold:
                state_2 = 2
            

            #棋子3
            cheese_3_x = center_x + (width/3)*(math.cos(angle_rad)) - (height/3)*(math.sin(angle_rad))
            cheese_3_y = center_y + (width/3)*(math.sin(angle_rad)) + (height/3)*(math.cos(angle_rad))
            cheese_3_x_l ,cheese_3_x_r = int (cheese_3_x)-22 ,int(cheese_3_x+22) #roi用
            cheese_3_y_l ,cheese_3_y_r= int (cheese_3_y)-22 ,int(cheese_3_y)+22
            roi_3_b = mask_black[cheese_3_y_l:cheese_3_y_r,cheese_3_x_l:cheese_3_x_r] #反二值化
            roi_3_blue = mask_blue[cheese_3_y_l:cheese_3_y_r,cheese_3_x_l:cheese_3_x_r]
            roi_3_white = mask_white[cheese_3_y_l:cheese_3_y_r,cheese_3_x_l:cheese_3_x_r]
            if cv2.countNonZero(roi_3_blue)>threshold:
                state_3 = 0
            if cv2.countNonZero(roi_3_white)>threshold:
                state_3 = 1
            if cv2.countNonZero(roi_3_b)>threshold:
                state_3 = 2
            

            #棋子4
            cheese_4_x = center_x - (width/3)*(math.cos(angle_rad)) 
            cheese_4_y = center_y -(width/3)*(math.sin(angle_rad))
            cheese_4_x_l ,cheese_4_x_r = int (cheese_4_x)-22 ,int(cheese_4_x+22) #roi用
            cheese_4_y_l ,cheese_4_y_r= int (cheese_4_y)-22 ,int(cheese_4_y)+22
            roi_4_b = mask_black[cheese_4_y_l:cheese_4_y_r,cheese_4_x_l:cheese_4_x_r] #反二值化
            roi_4_blue = mask_blue[cheese_4_y_l:cheese_4_y_r,cheese_4_x_l:cheese_4_x_r]
            roi_4_white = mask_white[cheese_4_y_l:cheese_4_y_r,cheese_4_x_l:cheese_4_x_r]
            if cv2.countNonZero(roi_4_blue)>threshold:
                state_4 = 0
            if cv2.countNonZero(roi_4_white)>threshold:
                state_4 = 1
            if cv2.countNonZero(roi_4_b)>threshold:
                state_4 = 2
            


            #棋子5
            cheese_5_x = center_x
            cheese_5_y = center_y
            cheese_5_x_l ,cheese_5_x_r = int (cheese_5_x)-22 ,int(cheese_5_x+22) #roi用
            cheese_5_y_l ,cheese_5_y_r= int (cheese_5_y)-22 ,int(cheese_5_y)+22
            roi_5_b = mask_black[cheese_5_y_l:cheese_5_y_r,cheese_5_x_l:cheese_5_x_r] #反二值化
            roi_5_blue = mask_blue[cheese_5_y_l:cheese_5_y_r,cheese_5_x_l:cheese_5_x_r]
            roi_5_white = mask_white[cheese_4_y_l:cheese_5_y_r,cheese_5_x_l:cheese_5_x_r]
            if cv2.countNonZero(roi_5_blue)>threshold:
                state_5 = 0
            if cv2.countNonZero(roi_5_white)>threshold:
                state_5 = 1
            if cv2.countNonZero(roi_5_b)>threshold:
                state_5 = 2
            
            #棋子6
            cheese_6_x = center_x + (width/3)*(math.cos(angle_rad))
            cheese_6_y = center_y + (width/3)*(math.sin(angle_rad))
            cheese_6_x_l ,cheese_6_x_r = int (cheese_6_x)-22 ,int(cheese_6_x+22) #roi用
            cheese_6_y_l ,cheese_6_y_r= int (cheese_6_y)-22 ,int(cheese_6_y)+22
            roi_6_b = mask_black[cheese_6_y_l:cheese_6_y_r,cheese_6_x_l:cheese_6_x_r] #反二值化
            roi_6_blue = mask_blue[cheese_6_y_l:cheese_6_y_r,cheese_6_x_l:cheese_6_x_r]
            roi_6_white = mask_white[cheese_6_y_l:cheese_6_y_r,cheese_6_x_l:cheese_6_x_r]
            if cv2.countNonZero(roi_6_blue)>threshold:
                state_6 = 0
            if cv2.countNonZero(roi_6_white)>threshold:
                state_6 = 1
            if cv2.countNonZero(roi_6_b)>threshold:
                state_6 = 2
            

            #棋子7
            cheese_7_x = center_x - (width/3)*(math.cos(angle_rad)) +  (height/3)*(math.sin(angle_rad))
            cheese_7_y = center_y - (width/3)*(math.sin(angle_rad)) -  (height/3)*(math.cos(angle_rad))
            cheese_7_x_l ,cheese_7_x_r = int (cheese_7_x)-22 ,int(cheese_7_x+22) #roi用
            cheese_7_y_l ,cheese_7_y_r= int (cheese_7_y)-22 ,int(cheese_7_y)+22
            roi_7_b = mask_black[cheese_7_y_l:cheese_7_y_r,cheese_7_x_l:cheese_7_x_r] #反二值化
            roi_7_blue = mask_blue[cheese_7_y_l:cheese_7_y_r,cheese_7_x_l:cheese_7_x_r]
            roi_7_white = mask_white[cheese_7_y_l:cheese_7_y_r,cheese_7_x_l:cheese_7_x_r]
            if cv2.countNonZero(roi_7_blue)>threshold:
                state_7 = 0
            if cv2.countNonZero(roi_7_white)>threshold:
                state_7 = 1
            if cv2.countNonZero(roi_7_b)>threshold:
                state_7 = 2
            
            #棋子8
            cheese_8_x = center_x + (height/3)*(math.sin(angle_rad))
            cheese_8_y = center_y - (height/3)*(math.cos(angle_rad)) 
            cheese_8_x_l ,cheese_8_x_r = int (cheese_8_x)-22 ,int(cheese_8_x+22) #roi用
            cheese_8_y_l ,cheese_8_y_r= int (cheese_8_y)-22 ,int(cheese_8_y)+22
            roi_8_b = mask_black[cheese_8_y_l:cheese_8_y_r,cheese_8_x_l:cheese_8_x_r] #反二值化
            roi_8_blue = mask_blue[cheese_8_y_l:cheese_8_y_r,cheese_8_x_l:cheese_8_x_r]
            roi_8_white = mask_white[cheese_8_y_l:cheese_8_y_r,cheese_8_x_l:cheese_8_x_r]
            if cv2.countNonZero(roi_8_blue)>threshold:
                state_8 = 0
            if cv2.countNonZero(roi_8_white)>threshold:
                state_8 = 1
            if cv2.countNonZero(roi_8_b)>threshold:
                state_8 = 2
        
            #棋子9
            cheese_9_x = center_x + (width/3)*(math.cos(angle_rad)) +  (height/3)*(math.sin(angle_rad))
            cheese_9_y = center_y + (width/3)*(math.sin(angle_rad)) -  (height/3)*(math.cos(angle_rad))
            cheese_9_x_l ,cheese_9_x_r = int (cheese_9_x)-22 ,int(cheese_9_x+22) #roi用
            cheese_9_y_l ,cheese_9_y_r= int (cheese_9_y)-22 ,int(cheese_9_y)+22
            roi_9_b = mask_black[cheese_9_y_l:cheese_9_y_r,cheese_9_x_l:cheese_9_x_r] #反二值化
            roi_9_blue = mask_blue[cheese_9_y_l:cheese_9_y_r,cheese_9_x_l:cheese_9_x_r]
            roi_9_white = mask_white[cheese_9_y_l:cheese_9_y_r,cheese_9_x_l:cheese_9_x_r]
            if cv2.countNonZero(roi_9_blue)>threshold:
                state_9 = 0
            if cv2.countNonZero(roi_9_white)>threshold:
                state_9 = 1
            if cv2.countNonZero(roi_9_b)>threshold:
                state_9 = 2
            
            board[0][0] = state_1
            board[0][1] = state_4
            board[0][2] = state_7
            board[1][0] = state_2
            board[1][1] = state_5
            board[1][2] = state_8
            board[2][0] = state_3
            board[2][1] = state_6
            board[2][2] = state_9
            
            #备下棋子检验
            global board_black,board_white,black_1_x,black_1_y,black_2_x,black_2_y,black_3_x,black_3_y,black_4_x,black_4_y,black_5_x,black_5_y
            #黑棋1
            black_1_x = cheese_1_x-(cheese_5_x-cheese_1_x)
            black_1_y = cheese_1_y-(cheese_5_y-cheese_1_y)
            black_1_x_l ,black_1_x_r = int(black_1_x)-22 , int(black_1_x)+22
            black_1_y_l ,black_1_y_r = int(black_1_y)-22 , int(black_1_y)+22
            roi_cheese_1_black = mask_black[black_1_y_l:black_1_y_r,black_1_x_l:black_1_x_r]
            if cv2.countNonZero(roi_cheese_1_black)>threshold:
                black_1_state = 1
            else:
                black_1_state = 0
            
            #黑棋2
            black_2_x = cheese_1_x-(cheese_4_x-cheese_1_x)
            black_2_y = cheese_1_y-(cheese_4_y-cheese_1_y)
            black_2_x_l ,black_2_x_r = int(black_2_x)-22 , int(black_2_x)+22
            black_2_y_l ,black_2_y_r = int(black_2_y)-22 , int(black_2_y)+22
            roi_cheese_2_black = mask_black[black_2_y_l:black_2_y_r,black_2_x_l:black_2_x_r]
            if cv2.countNonZero(roi_cheese_2_black)>threshold:
                black_2_state = 1
            else:
                black_2_state = 0
            

            #黑棋3
            black_3_x = cheese_2_x-(cheese_5_x-cheese_2_x)
            black_3_y = cheese_2_y-(cheese_5_y-cheese_2_y)
            black_3_x_l ,black_3_x_r = int(black_3_x)-22 , int(black_3_x)+22
            black_3_y_l ,black_3_y_r = int(black_3_y)-22 , int(black_3_y)+22
            roi_cheese_3_black = mask_black[black_3_y_l:black_3_y_r,black_3_x_l:black_3_x_r]
            if cv2.countNonZero(roi_cheese_3_black)>threshold:
                black_3_state = 1
            else:
                black_3_state = 0

            #黑棋4
            black_4_x = cheese_3_x-(cheese_6_x-cheese_3_x)
            black_4_y = cheese_3_y-(cheese_6_y-cheese_3_y)
            black_4_x_l ,black_4_x_r = int(black_4_x)-22 , int(black_4_x)+22
            black_4_y_l ,black_4_y_r = int(black_4_y)-22 , int(black_4_y)+22
            roi_cheese_4_black = mask_black[black_4_y_l:black_4_y_r,black_4_x_l:black_4_x_r]
            if cv2.countNonZero(roi_cheese_4_black)>threshold:
                black_4_state = 1
            else:
                black_4_state = 0

            #黑棋5
            black_5_x = cheese_3_x-(cheese_5_x-cheese_3_x)
            black_5_y = cheese_3_y+(cheese_3_y-cheese_5_y)
            black_5_x_l ,black_5_x_r = int(black_5_x)-22 , int(black_5_x)+22
            black_5_y_l ,black_5_y_r = int(black_5_y)-22 , int(black_5_y)+22
            roi_cheese_5_black = mask_black[black_5_y_l:black_5_y_r,black_5_x_l:black_5_x_r]
            if cv2.countNonZero(roi_cheese_5_black)>threshold:
                black_5_state = 1
            else:
                black_5_state = 0
            board_black[0] = black_1_state
            board_black[1] = black_2_state
            board_black[2] = black_3_state
            board_black[3] = black_4_state
            board_black[4] = black_5_state
            
            global white_1_x,white_1_y,white_2_x,white_2_y,white_3_x,white_3_y,white_4_x,white_4_y,white_4_x,white_4_y,white_5_x,white_5_y            
            #白棋1
            white_1_x = cheese_7_x+(cheese_7_x-cheese_5_x)
            white_1_y = cheese_7_y-(cheese_5_y-cheese_7_y)
            white_1_x_l ,white_1_x_r = int(white_1_x)-22 , int(white_1_x)+22
            white_1_y_l ,white_1_y_r = int(white_1_y)-22 , int(white_1_y)+22
            roi_cheese_1_white = mask_white[white_1_y_l:white_1_y_r,white_1_x_l:white_1_x_r]
            if cv2.countNonZero(roi_cheese_1_white)>threshold:
                white_1_state = 1
            else:
                white_1_state = 0

            #白棋2
            white_2_x = cheese_7_x+(cheese_7_x-cheese_4_x)
            white_2_y = cheese_7_y+(cheese_7_y-cheese_4_y)
            white_2_x_l ,white_2_x_r = int(white_2_x)-22 , int(white_2_x)+22
            white_2_y_l ,white_2_y_r = int(white_2_y)-22 , int(white_2_y)+22
            roi_cheese_2_white = mask_white[white_2_y_l:white_2_y_r,white_2_x_l:white_2_x_r]
            if cv2.countNonZero(roi_cheese_2_white)>threshold:
                white_2_state = 1
            else:
                white_2_state = 0
            #白棋3
            white_3_x = cheese_8_x+(cheese_8_x-cheese_5_x)
            white_3_y = cheese_8_y+(cheese_8_y-cheese_5_y)
            white_3_x_l ,white_3_x_r = int(white_3_x)-22 , int(white_3_x)+22
            white_3_y_l ,white_3_y_r = int(white_3_y)-22 , int(white_3_y)+22
            roi_cheese_3_white = mask_white[white_3_y_l:white_3_y_r,white_3_x_l:white_3_x_r]
            if cv2.countNonZero(roi_cheese_3_white)>threshold:
                white_3_state = 1
            else:
                white_3_state = 0
            
            #白棋4
            white_4_x = cheese_9_x+(cheese_9_x-cheese_6_x)
            white_4_y = cheese_9_y+(cheese_9_y-cheese_6_y)
            white_4_x_l ,white_4_x_r = int(white_4_x)-22 , int(white_4_x)+22
            white_4_y_l ,white_4_y_r = int(white_4_y)-22 , int(white_4_y)+22
            roi_cheese_4_white = mask_white[white_4_y_l:white_4_y_r,white_4_x_l:white_4_x_r]
            if cv2.countNonZero(roi_cheese_4_white)>threshold:
                white_4_state = 1
            else:
                white_4_state = 0
            
            #白棋5
            white_5_x = cheese_9_x+(cheese_9_x-cheese_5_x)
            white_5_y = cheese_9_y+(cheese_9_y-cheese_5_y)
            white_5_x_l ,white_5_x_r = int(white_5_x)-22 , int(white_5_x)+22
            white_5_y_l ,white_5_y_r = int(white_5_y)-22 , int(white_5_y)+22
            roi_cheese_5_white = mask_white[white_5_y_l:white_5_y_r,white_5_x_l:white_5_x_r]
            if cv2.countNonZero(roi_cheese_5_white)>threshold:
                white_5_state = 1
            else:
                white_5_state = 0
            board_white[0] = white_1_state
            board_white[1] = white_2_state
            board_white[2] = white_3_state
            board_white[3] = white_4_state
            board_white[4] = white_5_state
            cv2.rectangle(frame,(int(black_5_x-2),int(black_5_y-2)),(int(black_5_x+2),int(black_5_y+2)),(0,0,0),5)
            cv2.rectangle(frame,(int(black_4_x-2),int(black_4_y-2)),(int(black_4_x+2),int(black_4_y+2)),(0,0,0),5)
            cv2.rectangle(frame,(int(black_3_x-2),int(black_3_y-2)),(int(black_3_x+2),int(black_3_y+2)),(0,0,0),5)
            cv2.rectangle(frame,(int(black_2_x-2),int(black_2_y-2)),(int(black_2_x+2),int(black_2_y+2)),(0,0,0),5)
            cv2.rectangle(frame,(int(black_1_x-2),int(black_1_y-2)),(int(black_1_x+2),int(black_1_y+2)),(0,0,0),5)
            cv2.rectangle(frame,(int(white_1_x-2),int(white_1_y-2)),(int(white_1_x+2),int(white_1_y+2)),(255,255,255),5)
            cv2.rectangle(frame,(int(white_2_x-2),int(white_2_y-2)),(int(white_2_x+2),int(white_2_y+2)),(255,255,255),5)
            cv2.rectangle(frame,(int(white_3_x-2),int(white_3_y-2)),(int(white_3_x+2),int(white_3_y+2)),(255,255,255),5)
            cv2.rectangle(frame,(int(white_4_x-2),int(white_4_y-2)),(int(white_4_x+2),int(white_4_y+2)),(255,255,255),5)
            cv2.rectangle(frame,(int(white_5_x-2),int(white_5_y-2)),(int(white_5_x+2),int(white_5_y+2)),(255,255,255),5)
            cv2.rectangle(frame,(int(cheese_1_x-2),int(cheese_1_y-2)),(int(cheese_1_x+2),int(cheese_1_y+2)),(255,255,255),5)
            cv2.rectangle(frame,(int(cheese_9_x-2),int(cheese_9_y-2)),(int(cheese_9_x+2),int(cheese_9_y+2)),(255,255,255),5)
            cv2.rectangle(frame,(int(cheese_5_x-2),int(cheese_5_y-2)),(int(cheese_5_x+2),int(cheese_5_y+2)),(255,255,255),5)
            global k_real_frame
            k_real_frame = 19.6/(width+height)
            # 在图像上画出相关信息，方便调试
            if not abs(width - height)<100:
                board[0][0] = -1
            if flag == 0:
                board[0][0] = -1 
    cv2.waitKey(1)    


def read_buffer():
    for i in range(100):
        cap.grab() 
        cv2.waitKey(1)


def check_board():
    global board_steps
    global board
    if len(board_steps) == 0:
        while True:
            read_buffer()           
            req_board()
            cur_board = copy.deepcopy(board)
            if cur_board[0][0] != -1:
                board_steps.append(cur_board)
                print("Board Init")
                return cur_board

    while True:
        read_buffer()
        req_board()
        cur_board = copy.deepcopy(board)


        print("board_steps")
        print(board_steps[-1])
        print(board)
        if cur_board[0][0] == -1:
            print("No board!")
            continue
        if (cur_board == board_steps[-1]):
            print("No change done")
            continue
        match_flag = 1
        i = 0
        for i in range(20):
            req_board()
            
            if cur_board != board:
                match_flag = 0
            if board[0][0] == -1:
                match_flag = 0
            time.sleep(0.1)
            
        if match_flag == 1:
            break
    
    print("Board Update!")           
    board_steps.append(cur_board)               #小心添加指针
    return cur_board


def computer_first(num):
    board_steps.append([[0,0,0],[0,0,0],[0,0,0]])
    board = board_steps[-1]
    #print_board(board)####
    current_position =[0,0]

    from_p2p(10,num)
    time.sleep(1)
    from_p2p(10,num)
    board[num%3 - 1][num // 3] = 2
    board_steps.append(board)
    print(board)
    time.sleep(23)

    
    while True:
        
        cmd_go(0,0)
        cheese_sleep_time = calculate_time([0,0],[0,0],current_position)
        print("goto sero time")
        print(cheese_sleep_time+1)
        time.sleep(cheese_sleep_time+1)
        # while True:
        #     if com.in_waiting > 0 :
        #         data_arm = com.read(1)#从机械臂收回放置完成指令
        #         received_byte = int.from_bytes(data_arm, byteorder='little')
        #         if received_byte == 0x99:
        #             print("GO 0 Done!")
        #             break
        check_board()
        board = board_steps[-1]
        #print_board(board)####
        if check_winner(board, 1):
            print("You win!")
            time.sleep(5)
            str_ready = b't0.txt=\"you win\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break
        if not get_empty_cells(board):
            print("It's a tie!")
            time.sleep(5)
            str_ready = b't0.txt=\"tie game\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break

        current_position = computer_move_cf(board)
        
        print("Computer's move:")
        print_board(board)
        if check_winner(board, 2):
            print("Computer wins!")
            time.sleep(5)
            str_ready = b't0.txt=\"you lose\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break
        if not get_empty_cells(board):
            print("It's a tie!")
            time.sleep(5)
            str_ready = b't0.txt=\"tie game\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break

def user_first():
    board = board_steps[-1]
    #print_board(board)####
    current_position =[0,0]
    while True:
        
        cmd_go(0,0)
        cheese_sleep_time = calculate_time([0,0],[0,0],current_position)
        print("goto sero time")
        print(cheese_sleep_time+1)
        time.sleep(cheese_sleep_time+1)
        # while True:
        #     if com.in_waiting > 0 :
        #         data_arm = com.read(1)#从机械臂收回放置完成指令
        #         received_byte = int.from_bytes(data_arm, byteorder='little')
        #         if received_byte == 0x99:
        #             print("GO 0 Done!")
        #             break
        check_board()
        board = board_steps[-1]
        #print_board(board)####
        if check_winner(board, 2):
            print("You win!")
            time.sleep(5)
            str_ready = b't0.txt=\"you win\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break
        if not get_empty_cells(board):
            print("It's a tie!")
            time.sleep(5)
            str_ready = b't0.txt=\"tie game\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break

        current_position = computer_move(board)
        
        print("Computer's move:")
        print_board(board)
        if check_winner(board, 1):
            print("Computer wins!")
            time.sleep(5)
            str_ready = b't0.txt=\"you lose\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break
        if not get_empty_cells(board):
            print("It's a tie!")
            time.sleep(5)
            str_ready = b't0.txt=\"tie game\"'
            data_fin = b'\xff\xff\xff'
            data = str_ready + data_fin
            com_HMI.write(data)
            break

# def enval_global():
#     abs_1_x,abs_1_y =  
#     abs_2_x,abs_2_y = 
#     abs_3_x,abs_3_y = 
#     abs_4_x,abs_4_y = 
#     abs_5_x,abs_5_y = 
#     abs_6_x,abs_6_y = 
#     abs_7_x,abs_7_y = 
#     abs_8_x,abs_8_y = 
#     abs_9_x,abs_9_y = 
#     abs_b_1_x,abs_b_1_y = 
#     abs_b_2_x,abs_b_2_y = 
#     abs_b_3_x,abs_b_3_y = 
#     abs_b_4_x,abs_b_4_y =
#     abs_b_5_x,abs_b_5_y = 
#     abs_w_1_x,abs_w_1_y = 
#     abs_w_2_x,abs_w_2_y = 
#     abs_w_3_x,abs_w_3_y = 
#     abs_w_4_x,abs_w_4_y = 
#     abs_w_5_x,abs_w_5_y = 
    

def rotate_point(abs_x, abs_y):
    
    return [abs_x, abs_y]

def get_white_location(data):
 #   global angle_rad,abs_1_x,abs_2_x,abs_3_x,abs_4_x,abs_5_x,abs_6_x,abs_7_x,abs_8_x,abs_9_x,abs_1_y,abs_2_y,abs_3_y,abs_4_y,abs_5_y,abs_6_y,abs_7_y,abs_8_y,abs_9_y,abs_b_1_x,abs_b_2_x,abs_b_3_x,abs_b_4_x,abs_b_5_x,abs_b_1_y,abs_b_2_y,abs_b_3_y,abs_b_4_y,abs_b_5_y,abs_w_1_x,abs_w_2_x,abs_w_3_x,abs_w_4_x,abs_w_5_x,abs_w_1_y,abs_w_2_y,abs_w_3_y,abs_w_4_y,abs_w_5_y = 0
    # angle_rad = 0
    # if data[2] == 0x01:
    #     location = rotate_point(abs_w_1_x,abs_w_1_y,angle_rad)
    #     return location
    # elif data[2] == 0x02:
    #     location = rotate_point(abs_w_2_x,abs_w_2_y,angle_rad)
    #     return location
    # elif data[2] == 0x03:
    #     location = rotate_point(abs_w_3_x,abs_w_3_y,angle_rad)
    #     return location
    # elif data[2] == 0x04:
    #     location = rotate_point(abs_w_4_x,abs_w_4_y,angle_rad)
    #     return location
    # elif data[2] == 0x05:
    #     location = rotate_point(abs_w_5_x,abs_w_5_y,angle_rad)
    #     return location
    # else:
    #     return [-1,-1]
        location = [Positions[int(data[2])+13][0],Positions[int(data[2])+13][1]]
        return location
    
    
def get_black_location(data): 
        # angle_rad =0
        # if data[2] == 0x01:
        #     location = rotate_point(abs_b_1_x,abs_b_1_y,angle_rad)
        #     return location
        # elif data[2] == 0x02:
        #     location = rotate_point(abs_b_2_x,abs_b_2_y,angle_rad)
        #     return location
        # elif data[2] == 0x03:
        #     location = rotate_point(abs_b_3_x,abs_b_3_y,angle_rad)
        #     return location
        # elif data[2] == 0x04:
        #     location = rotate_point(abs_b_4_x,abs_b_4_y,angle_rad)
        #     return location
        # elif data[2] == 0x05:
        #     location = rotate_point(abs_b_5_x,abs_b_5_y,angle_rad)
        #     return location
        # else:
        #     return [-1,-1]
        
        location = [Positions[int(data[2])+8][0],Positions[int(data[2])+8][1]]
        return location


def get_cheese_location(data):     
            location = rotate_point(Positions[int(data[2])][0],Positions[int(data[2])][1])
            return location

            # if data[2] == 0x01:
            #     location = rotate_point(abs_1_x,abs_1_y,angle_rad)
            #     return location
            # elif data[2] == 0x02:
            #     location = rotate_point(abs_2_x,abs_2_y,angle_rad)
            #     return location
            # elif data[2] == 0x03:
            #     location = rotate_point(abs_3_x,abs_3_y,angle_rad)
            #     return location
            # elif data[2] == 0x04:
            #     location = rotate_point(abs_4_x,abs_4_y,angle_rad)
            #     return location
            # elif data[2] == 0x05:
            #     location = rotate_point(abs_5_x,abs_5_y,angle_rad)
            #     return location
            # elif data[2] == 0x06:
            #     location = rotate_point(abs_6_x,abs_6_y,angle_rad)
            #     return location
            # elif data[2] == 0x07:
            #     location = rotate_point(abs_7_x,abs_7_y,angle_rad)
            #     return location
            # elif data[2] == 0x08:
            #     location = rotate_point(abs_8_x,abs_8_y,angle_rad)
            #     return location
            # elif data[2] == 0x09:
            #     location = rotate_point(abs_9_x,abs_9_y,angle_rad)
            #     return location
            # else:
            #     return [-1,-1]

def cheese_position_init():
    
    if board_white == [1,1,1,1,1] and board_black == [1,1,1,1,1]:
        str_ready = b't0.txt=\"Ready\"'
        data_fin = b'\xff\xff\xff'
        data = str_ready + data_fin
        com_HMI.write(data)
        return 1
    else: 
        return 0

def cheese_choice(data):
    if data[1] ==0x01 :#选择黑色棋子
        location = get_black_location(data)
    elif data[1] == 0x02 :#选择白色
        location = get_white_location(data)
    while True:
        send_flag = 0
        if com_HMI.in_waiting > 0 :
            data = com_HMI.read(6)
            if data[0] == 0x55 and data[1] == 0x01:
                print(int(location[0]),int(location[1]),int(Positions[int(data[2])-1][0]),int(Positions[int(data[2])-1][1]))
                cmd_move(int(location[0]),int(location[1]),int(Positions[int(data[2])-1][0]),int(Positions[int(data[2])-1][1]))
                send_flag = 1
                str_ready = b't0.txt=\"planting...\"'
                data_fin = b'\xff\xff\xff'
                data = str_ready + data_fin
                com_HMI.write(data)    
                break

        # while send_flag :
        #     data_arm = [0x25]
        #     data_arm[0] = com.read(2)#从机械臂收回放置完成指令
        #     print(data_arm)
        #     if data_arm[0] == b'\x99':
        #         print(data_arm)
        #         cmd_go(0,0)
        #         send_flag = 0
        #         return True

     

def cheese_position_set(data):

    #选择第一个棋子和第一个坐标
    global location_start_1
    while not com_HMI.in_waiting > 0 :
        abc = 0
    data = com_HMI.read(6)#选择第一个棋子的位置
    if data[0] == 0x44 and data[1] == 0x01:
        location_start_1 = get_black_location(data)
    elif data[0] == 0x44 and data[1] == 0x02:
        location_start_1 = get_white_location(data)
    while not com_HMI.in_waiting > 0 :
          abc = 0
    data = com_HMI.read(6)#选择第一个棋子的目标位置
    if data[0] == 0x55 and data[1] == 0x01:
        location_target_1 = Positions[int(data[2])-1]


    #选择第二个棋子和第二个坐标
    while not com_HMI.in_waiting > 0 :
        abc = 0
    data = com_HMI.read(6)#选择第二个棋子的位置
    if data[0] == 0x44 and data[1] == 0x01:
        location_start_2 = get_black_location(data)
    elif data[0] == 0x44 and data[1] == 0x02:
        location_start_2 = get_white_location(data)
    while not com_HMI.in_waiting > 0 :
          abc = 0
    data = com_HMI.read(6)#选择第二个棋子的目标位置
    if data[0] == 0x55 and data[1] == 0x01:
        location_target_2 = Positions[int(data[2])-1]

    #选择第三个棋子位置
    while not com_HMI.in_waiting > 0 :
        abc = 0
    data = com_HMI.read(6)#选择第二个棋子的位置
    if data[0] == 0x44 and data[1] == 0x01:
        location_start_3 = get_black_location(data)
    elif data[0] == 0x44 and data[1] == 0x02:
        location_start_3 = get_white_location(data)
    while not com_HMI.in_waiting > 0 :
          abc = 0
    data = com_HMI.read(6)#选择第二个棋子的目标位置
    if data[0] == 0x55 and data[1] == 0x01:
        location_target_3 = Positions[int(data[2])-1]

    # 选择第四个棋子位置
    while not com_HMI.in_waiting > 0:
        abc = 0
    data = com_HMI.read(6)  # 选择第二个棋子的位置
    if data[0] == 0x44 and data[1] == 0x01:
        location_start_4 = get_black_location(data)
    elif data[0] == 0x44 and data[1] == 0x02:
        location_start_4 = get_white_location(data)
    while not com_HMI.in_waiting > 0:
        abc = 0
    data = com_HMI.read(6)  # 选择第二个棋子的目标位置
    if data[0] == 0x55 and data[1] == 0x01:
        location_target_4 = Positions[int(data[2]) - 1]
    time.sleep(1)
    sleep_time_plus = 14           ##################参数修改
    cmd_move(int(location_start_1[0]), int(location_start_1[1]), int(location_target_1[0]),int(location_target_1[1]))
    sleep_time_1 = calculate_time([0,0],location_start_1,location_target_1)+sleep_time_plus
    time.sleep(sleep_time_1)
    print(int(location_start_1[0]), int(location_start_1[1]), int(location_target_1[0]),int(location_target_1[1]))
    print(sleep_time_1)

    cmd_move(int(location_start_2[0]), int(location_start_2[1]), int(location_target_2[0]),int(location_target_2[1]))
    sleep_time_2 = calculate_time([0,0],location_start_2,location_target_1)+sleep_time_plus
    time.sleep(sleep_time_2)
    print(int(location_start_2[0]), int(location_start_2[1]), int(location_target_2[0]),int(location_target_2[1]))
    print(sleep_time_2)

    cmd_move(int(location_start_3[0]), int(location_start_3[1]), int(location_target_3[0]),int(location_target_3[1]))
    sleep_time_3 = calculate_time([0,0],location_start_3,location_target_3)+sleep_time_plus
    time.sleep(sleep_time_3)
    print(int(location_start_3[0]), int(location_start_3[1]), int(location_target_3[0]),int(location_target_3[1]))
    print(sleep_time_3)

    cmd_move(int(location_start_4[0]), int(location_start_4[1]), int(location_target_4[0]),int(location_target_4[1]))
    sleep_time_4 = calculate_time([0,0],location_start_4,location_target_4)+sleep_time_plus
    time.sleep(sleep_time_4)
    print(int(location_start_4[0]), int(location_start_4[1]), int(location_target_4[0]),int(location_target_4[1]))
    print(sleep_time_4)





def calculate_time(location_pre,location_start,location_target):
    distance_mm = abs(location_pre[0]-location_start[0])+abs(location_start[0]-location_target[0])+abs(location_pre[1]-location_start[1])+abs(location_start[1]-location_target[1])
    pulse = distance_mm * 100
    exe_time = 1000 * (pulse) / (3200 * 60 / 60)
    return exe_time/1000
        
def cmd_go(x, y):
    data_head = b'\x42'
    data_cmd = b'\x50'
    data_load1 = x.to_bytes(2, byteorder = 'little')
    data_load2 = y.to_bytes(2, byteorder = 'little')
    data_end = b'\x00\x00\x00\x00'
    data = data_head + data_cmd + data_load1 + data_load2 + data_end
    com.write(data)


def cmd_move(x1, y1, x2, y2):
    data_head = b'\x42'
    data_cmd = b'\x53'
    data_load1 = x1.to_bytes(2, byteorder = 'little')
    data_load2 = y1.to_bytes(2, byteorder = 'little')
    data_load3 = x2.to_bytes(2, byteorder = 'little')
    data_load4 = y2.to_bytes(2, byteorder = 'little')
    data = data_head + data_cmd + data_load1 + data_load2 + data_load3 + data_load4
    com.write(data)

# def rotate_point(abs_x, abs_y, angle_rad):
    
#     # 计算旋转后的坐标
#     x_new = abs_x * math.cos(angle_rad) - abs_y * math.sin(angle_rad)
#     y_new = abs_x * math.sin(angle_rad) + abs_y * math.cos(angle_rad)
    
#     return x_new, y_new

x = 0
y = 0

def from_p2p(num1,num2):
    position1 = Positions[num1-1]
    position2 = Positions[num2-1]
    cmd_move(position1[0],position1[1],position2[0],position2[1])
    
    # while True:##########################################可能改为time.sleep(20)
    #     if com.in_waiting > 0 :
    #         data_arm = com.read(1)#从机械臂收回放置完成指令
    #         received_byte = int.from_bytes(data_arm, byteorder='little')
    #         if received_byte == 0x99:
    #             cmd_go(0,0)
    #             print("P2P_1 Done!")
    #             break


if __name__ == "__main__":
    #cmd_go(60,30)
    
    
    while True:
        req_board()
        print(board)
        cv2.imshow('result',frame)           
        cv2.imshow('blue',mask_blue)
        cv2.imshow('white',mask_white)
        cv2.imshow("black",mask_black)
        
        
        
        # if cheese_position_init()==1:     
        #     print("continue")
        
        if com_HMI.in_waiting > 0 :
            data = com_HMI.read(6)
            if data[0] == 0x33:
                parameter_adjustment(data)
            elif data[0] == 0x44:
                cheese_choice(data)
            elif data[0] == 0x55:
                cheese_position_set(data)
            elif data[0] == 0x66:
                while not com_HMI.in_waiting>0:
                    abc = 0                
                data = com_HMI.read(6)               
                cheese_num = int(data[2])                
                computer_first(cheese_num)
            elif data[0] == 0x77:
                user_first()
            elif data[0] == 0x42:
                if data[1] == 0x54:
                    cmd_go(0,0)
                    # data = b'\x42\x54\00\00\00\00\00\00\00\00'
                    # com.write(data)             

        cv2.waitKey(1)
            
        
    
# 释放资源
cap.release()
cv2.destroyAllWindows()
