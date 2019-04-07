# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:52:37 2019

判断点是否在多边形内部的api

@author: github:QLWeilcf
@modify: zyb_as
"""

# 点是否在外包矩形内
def isPointWithinBox(poi, sbox, toler=0.0001):
	# sbox=[[x1,y1],[x2,y2]]
	# 不考虑在边界上，需要考虑就加等号
	if poi[0] > sbox[0][0] and poi[0] < sbox[1][0] and poi[1] > sbox[0][1] and poi[1] < sbox[1][1]:
		return True
	if toler > 0:
		pass
	return False


def isRayIntersectsSegment(poi,s_poi,e_poi): #[x,y] [lng,lat]
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1]==e_poi[1]: #排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1]>poi[1] and e_poi[1]>poi[1]: #线段在射线上边
        return False
    if s_poi[1]<poi[1] and e_poi[1]<poi[1]: #线段在射线下边
        return False
    if s_poi[1]==poi[1] and e_poi[1]>poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1]==poi[1] and s_poi[1]>poi[1]: #交点为下端点，对应epoint
        return False
    if s_poi[0]<poi[0] and e_poi[0]<poi[0]: #线段在射线左边
        return False

    xseg=e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
    if xseg<poi[0]: #交点在射线起点的左侧
        return False
    return True  #排除上述情况之后


def isPointWithinPoly(point, poly):
    '''
    judging whether the point is in a restricted object's segmentation
    we use the ray method, see for more: 
    https://www.jianshu.com/p/ba03c600a557
    #poly= [ [[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]], ... ]
    '''
    #可以先判断点是否在外包矩形内 
    #if not isPoiWithinBox(point,mbr=[[0,0],[180,90]]): return False
    #但算最小外包矩形本身需要循环边，会造成开销，本处略去
    
    # 求与射线的交点个数
    sinsc=0 #交点个数
    for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
        for i in range(len(epoly)-1): #[0,len-1]
            s_poi=epoly[i]
            e_poi=epoly[i+1]
            if isRayIntersectsSegment(point,s_poi,e_poi):
                sinsc+=1 #有交点就加1
    # 交点个数为奇数，则在多边形内
    return True if sinsc%2==1 else  False









'''
# 凸四边形的计算方法，可扩展为判断点是否在凸多边形内部的方法
# 但是，无法处理任意多边形的情况
def distince(point1,point2):
    dist = math.sqrt(math.pow(point2[1]-point1[1] ,2) + math.pow(point2[0]-point1[0] ,2))
    return dist

def angle(point_main,point1,point2):
    dist_main1 = distince(point_main,point1)
    dist_main2 = distince(point_main,point2)
    dist_other = distince(point1,point2)
    cos_value = (math.pow(dist_other,2) - math.pow(dist_main1,2) - 
                 math.pow(dist_main2,2))/(-2 * dist_main1 * dist_main2)
    angel_ = round(math.degrees(math.acos(cos_value)),0)
    #print(angel_)
    return angel_

def isornoinbyangle(point1,point2,point3,point4,point_test):
    "note:point1 and point3 are on different sides of the rectangle "
    angle_1 = angle(point_test,point1,point2)
    angle_2 = angle(point_test,point1,point4)
    angle_3 = angle(point_test,point3,point4)
    angle_4 = angle(point_test,point3,point2)
    totol_angle = angle_1 + angle_2 + angle_3 + angle_4
    print(totol_angle)
    #print(math.degrees(math.acos(totol_cos)))
    if totol_angle < 360:
        return False
    else:
        return True
isornoinbyangle([1,1],[1,3],[3,3],[3,1],[3.1,3.0001]) 

def function(point1,point2):
    if point2[0] - point1[0] == 0:
        A = "kill"
        B = "kill"
        C = "kill"    
    else:
        A = (point2[1] - point1[1])/(point2[0] - point1[0])
        B = -1
        C = point1[1] - A * point1[0]
    return A,B,C
def point2line(point1,point2,point_main):
    A,B,C = function(point1,point2)
    if A == "kill":
        distince = abs(point_main[1] - point1[1])
    else:
        distince = abs(A * point_main[0] + B * point_main[1] + C) / math.sqrt(math.pow(A,2) + math.pow(B,2))
    return distince
def isornoinbydist(point1,point2,point3,point4,point_test):
    "note:point1 and point3 are on different sides of the rectangle "
    dist_1 = point2line(point1,point2,point_test)
    dist_2 = point2line(point1,point4,point_test)
    dist_3 = point2line(point3,point2,point_test)
    dist_4 = point2line(point3,point4,point_test)
    dist_1to2 = math.sqrt(math.pow((point2[1] - point1[1]),2) + 
                          math.pow((point2[0] - point1[0]),2))
    dist_1to4 = math.sqrt(math.pow((point4[1] - point1[1]),2) + 
                          math.pow((point4[0] - point1[0]),2))
    if dist_1 <= dist_1to4 and dist_4 <= dist_1to4 and dist_2 <= dist_1to2 and dist_3 <= dist_1to2:
        return True
    else:
        return False

isornoinbydist([1,1],[1,3],[3,3],[3,1],[2,3.1])  
'''
