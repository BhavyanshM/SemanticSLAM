import numpy as np

class RangeScanner:
    def __init__(self, max_range, num_points):
        self.max_range = max_range
        self.num_points = num_points

    def scan(self, pos, obstacles):
        points = []
        for i in range(self.num_points):
            theta = i * 2 * np.pi / self.num_points
            point = self.get_scan_point(pos, theta, obstacles)
            points.append(point)

            # print("i: {}, Theta: {}, Point: {}".format(i, theta, point))

        return points

    def get_scan_point(self, pos, theta, obstacles):
        # Get the end point of the ray
        end_point = np.array([pos[0] + self.max_range * np.cos(theta), pos[1] + self.max_range * np.sin(theta)])

        # Get the intersection point with the obstacles
        intersection_point = self.get_intersection_point(pos, end_point, obstacles)

        if np.linalg.norm(intersection_point - pos) > self.max_range:
            intersection_point = end_point

        return intersection_point
    
    def get_intersection_point(self, start_point, end_point, obstacles):

        # set closest_point to max, max
        closest_point = np.array([10000000, 10000000])

        for obstacle in obstacles:
            obstacle_center = obstacle[0:2]
            obstacle_size_x = obstacle[2]
            obstacle_size_y = obstacle[3]

            # Get all four corners of the obstacle
            obstacle_min_x = obstacle_center[0] - obstacle_size_x
            obstacle_max_x = obstacle_center[0] + obstacle_size_x
            obstacle_min_y = obstacle_center[1] - obstacle_size_y
            obstacle_max_y = obstacle_center[1] + obstacle_size_y

            # build polygon of obstacle with vertices
            polygon = np.array([[obstacle_min_x, obstacle_min_y], [obstacle_min_x, obstacle_max_y],
                                [obstacle_max_x, obstacle_max_y], [obstacle_max_x, obstacle_min_y]])
            

            # Get the closest intersection point of line segment with the obstacle polygon
            intersection_point = self.get_closest_intersection_point(start_point, end_point, polygon)

            # If the intersection point is closer than the current closest point, update the closest point
            if np.linalg.norm(intersection_point - start_point) < np.linalg.norm(closest_point - start_point):
                closest_point = intersection_point


        # check if intersection with simulation grid walls is closer than the current closest point
        intersection_point = self.get_wall_intersection_point(start_point, end_point, 
                                                            np.array([[0, 0], [0, 100], [100, 100], [100, 0]]))
        
        if np.linalg.norm(intersection_point - start_point) < np.linalg.norm(closest_point - start_point):
            closest_point = intersection_point

        return closest_point

    
    def get_closest_intersection_point(self, start_point, end_point, polygon):

        closest_point = np.array([10000000, 10000000])

        # find (m,c) for start-end line segment
        if end_point[0] - start_point[0] == 0:
            m1 = 0
        else:
            m1 = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        c1 = start_point[1] - m1 * start_point[0]

        for i in range(polygon.shape[0]):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % polygon.shape[0]]

            if p1[0] == p2[0]:
                y = m1 * p1[0] + c1
                x = p1[0]

                if y < min(p1[1], p2[1]) or y > max(p1[1], p2[1]):
                    continue

            elif p1[1] == p2[1] and m1 != 0:
                x = (p1[1] - c1) / m1
                y = p1[1]

                if x < min(p1[0], p2[0]) or x > max(p1[0], p2[0]):
                    continue

            point = np.array([x, y])

            if np.linalg.norm(point - start_point) < np.linalg.norm(closest_point - start_point):
                closest_point = point

        return closest_point
    
    def get_wall_intersection_point(self, start_point, end_point, polygon):

        good_point = np.array([10000000, 10000000])

        # find (m,c) for start-end line segment
        if end_point[0] - start_point[0] == 0:
            m1 = 0
        else:
            m1 = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        c1 = start_point[1] - m1 * start_point[0]

        for i in range(polygon.shape[0]):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % polygon.shape[0]]

            if p1[0] == p2[0]:
                y = m1 * p1[0] + c1
                x = p1[0]

                if y < min(p1[1], p2[1]) or y > max(p1[1], p2[1]):
                    continue

            elif p1[1] == p2[1] and m1 != 0:
                x = (p1[1] - c1) / m1
                y = p1[1]

                if x < min(p1[0], p2[0]) or x > max(p1[0], p2[0]):
                    continue

            point = np.array([x, y])


            ratio = (point[0] - end_point[0]) / (start_point[0] - end_point[0])

            if ratio < 0 or ratio > 1:
                continue

            good_point = point

        return good_point

    def is_point_inside_obstacle(self, point, obstacle):
        
        obstacle_min_x = obstacle[0] - obstacle[2]
        obstacle_max_x = obstacle[0] + obstacle[2]
        obstacle_min_y = obstacle[1] - obstacle[3]
        obstacle_max_y = obstacle[1] + obstacle[3]

        if point[0] < obstacle_min_x or point[0] > obstacle_max_x or point[1] < obstacle_min_y or point[1] > obstacle_max_y:
            return False
        
        return True


if __name__ == "__main__":
    scanner = RangeScanner(max_range=100, num_points=48)
    pos = np.array([30, 40])
    obstacles = [(15, 15, 3)]
    points = scanner.scan(pos, obstacles)
    
    import cv2

    # plot both the robot and the obstacles using pixel and rectangles

    # create a 100x100 pixel image
    img = np.zeros((101, 101, 3), np.uint8)

    # draw the robot
    img[int(pos[0]), int(pos[1])] = np.array([0, 255, 255])

    # draw the obstacles
    for obstacle in obstacles:
        cv2.rectangle(img, (int(obstacle[0] - obstacle[2]), int(obstacle[1] - obstacle[3])),
                      (int(obstacle[0] + obstacle[2]), int(obstacle[1] + obstacle[2])), (255, 255, 255), -1)
        
    # set the pixels in img at point to white
    for point in points:

        if int(point[0]) < img.shape[0] and int(point[1]) < img.shape[1] and int(point[0]) >= 0 and int(point[1]) >= 0:
            img[int(point[0]), int(point[1])] = np.array([0, 0, 255])

        
        
        
        

    # resize the window to make it bigger by a factor of 5
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1500, 1500)

    # show the image
    cv2.imshow("image", img)


    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
