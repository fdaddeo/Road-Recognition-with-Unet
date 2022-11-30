import os, argparse
import cv2
import numpy as np
from random import uniform
from requests import get as getResource
from bs4 import BeautifulSoup as bs
from json import loads, dump
from glob import glob
from multiprocessing import Process
from multiprocessing import cpu_count
from time import sleep
from tqdm import tqdm

KEY = '' # bing maps api key

RGB = 'rgb/'
MASK = 'mask/'
TXT_DATA = 'txt_data/'
JSON_DATA = 'json_data/'
MASK_POINT = 'mask_point/'
IMG_FORMAT = 'png'
DATA_FORMAT = 'json'

ZOOM = 18
WIDTH = 1500
HEIGHT = 1500

# For road width
WIDTH_A = 24  # red
WIDTH_B = 22 # green
WIDTH_C = 18 # white
WIDTH_D = 16 # blue

# Generates random number inside of city rectangle
def random_walk(top, left, bottom, right, max):
    to_ret = []
    for _ in range(max):
        lat = round(uniform(top, bottom), 6)
        lon = round(uniform(left, right), 6)
        to_ret.append((lat,lon))
    return to_ret 

class Image:
    def __init__(self, data_path, lat, lon, zoom, width, height,
        top = None, bottom = None, left = None, right = None, id = None,
        ways_info = [], ways = [], ways_pixels = [], ways_width = []):

        self.data_path = data_path
        self.center_lat = lat
        self.center_lon = lon
        self.zoom = zoom
        self.width = width
        self.height = height
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.id = id
        self.ways_info = ways_info # TODO: delete
        self.ways = ways
        self.ways_pixels = ways_pixels
        self.ways_width = ways_width

        if self.id is None:
            self.id = '{0}_{1}__{2}'.format(self.center_lat, self.center_lon, self.zoom)

    def get_corners(self):
        url = 'https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/{0},{1}/{2}?mapSize={3},{4}&key={5}&mapMetadata=1&o={6}'.format(
            self.center_lat, self.center_lon, self.zoom, self.width, self.height, KEY, DATA_FORMAT)

        html = getResource(url).text
        soup = bs(html, 'html.parser')
        content = loads(soup.text)
        box = content['resourceSets'][0]['resources'][0]['bbox']
        # box -> [S, W, N, E]
        self.top = box[2]
        self.bottom = box[0]
        self.left = box[1]
        self.right = box[3]

    # Gets each structural pixel of the road containe in the choosen rectangle
    def get_ways(self):

        ##### Getting xml repsonse #####
        #url = 'https://api.openstreetmap.org/api/0.6/map?bbox={0},{1},{2},{3}'.format(self.left, self.bottom, self.right, self.top)
        #url = 'http://overpass-api.de/api/interpreter?data=[out:xml];way({0},{1},{2},{3});(._;>;);out;'.format(self.bottom, self.left, self.top, self.right)
        url = 'http://overpass-api.de/api/interpreter?data=[out:xml];way({0},{1},{2},{3})[highway];(._;>;);out;'.format(self.bottom, self.left, self.top, self.right)

        
        xml = getResource(url).text
        soup = bs(xml, 'xml')

        # Check for response error
        # if xml == 'You have downloaded too much data. Please try again later.': # osm api response error
        if soup.find(text="Error"): # overpass api response error
            return False
            #print('Too much data, slowing down...')
            #sleep(2)

        # Getting nodes from xml repsonse
        node_list = [] # (id,lat,lon)
        nodes_tag = soup.find_all('node')
        for node_tag in nodes_tag:
            id = int(node_tag.attrs['id'])
            lat = float(node_tag.attrs['lat'])
            lon = float(node_tag.attrs['lon'])
            node_list.append((id, lat, lon))

        # Getting way points with lat,lon from xml repsonse and previously found nodes
        ref_list = [
            ('highway','motorway'), ('highway','trunk'), ('highway','primary'), ('highway','secondary'),
            ('highway','tertiary'), ('highway','unclassified'),('highway','residential'), 
            
            ('highway','motorway_link'), ('highway','trunk_link'), ('highway','primary_link'),
            ('highway','secondary_link'), ('highway','tertiary_link'),
            
            ('highway','living_street'), ('highway','service'), ('highway','pedestrian'),
            ('highway','bus_guideway'), ('highway','road'), ('highway','busway')]

        width_a = ['motorway']

        width_b = ['trunk','primary']
        
        width_c = ['secondary', 'tertiary']
        
        width_d = [
            'motorway_link', 'trunk_link', 'primary_link',
            'secondary_link', 'tertiary_link', 'unclassified',
            'residential','living_street', 'service',
            'pedestrian', 'bus_guideway', 'road', 'busway']

        # ref_list = [
        #     ('highway','motorway'), ('highway','trunk'), ('highway','primary'),
        #     ('highway','secondary'), ('highway','tertiary'),
        #     ('highway','motorway_link'), ('highway','trunk_link'), ('highway','primary_link'),
        #     ('highway','secondary_link'), ('highway','tertiary_link'),
        #     ('highway','living_street'), ('highway','service'),
        #     ('highway','bus_guideway'), ('highway','busway'), ('highway','steps'), 
        #     ('highway','unclassified'),
        #     ('highway','residential'), ('highway','pedestrian'), ('highway','track'),
        #     ('highway','road'), ('highway','footway'), ('highway','bridleway'),
        #     ('highway','path'), ('highway','cycleway')]
        # class_1 = ['primary', 'secondary', 'tertiary', 'residential', 'road']
        # class_2 = ['pedestrian', 'track', 'footway', 'bridleway', 'path', 'cycleway']
            
        way_list = [] # [(lat,lon)]
        ways_width = []

        ways = soup.find_all('way')
        for way in ways:
            tags_list = [] # (key,value)
            tags = way.find_all('tag')
            for tag in tags:
                k = tag.attrs['k']
                v = tag.attrs['v']
                tags_list.append((k,v))
            
            # if ref_list contains at least one element of tags_list
            # then get the nodes for this way
            if len(tags_list) > 0:
                if any(elem in ref_list for elem in tags_list):
                    ways_width_tmp = []
                    for t in tags_list:
                        if(t[1] in width_a):
                            ways_width_tmp.append(WIDTH_A)
                        elif(t[1] in width_b):
                            ways_width_tmp.append(WIDTH_B)
                        elif(t[1] in width_c):
                            ways_width_tmp.append(WIDTH_C)
                        elif(t[1] in width_d):
                            ways_width_tmp.append(WIDTH_D)
                        
                    ways_width.append(ways_width_tmp)
                        
                    nodes = [] # (lat,lon)
                    node_tags = way.find_all('nd')
                    for node_tag in node_tags:
                        id = int(node_tag.attrs['ref'])
                        for n in node_list:
                            if id == n[0]:
                                nodes.append((n[1], n[2]))

                    way_list.append(nodes)

        self.ways = way_list[:]
        self.ways_width = ways_width[:]

        if len(self.ways) != len(self.ways_width):
            print('ERROR WHILE PARSING RESPONSE')
            return False

        return True

    # Transform the absolute coordinates in pixel coorinates
    def get_ways_pixels(self):
        if self.ways:
            top = self.top
            bottom = self.bottom
            left = self.left
            right = self.right

            ways_pixels = []
            for way in self.ways:
                nodes_pixels = []
                for node in way:
                    y = int((HEIGHT - 1) - (HEIGHT * (node[0] - min(bottom, top)) / abs(bottom - top)))
                    x = int(WIDTH * (node[1] - min(left, right)) / abs(right - left))
                    nodes_pixels.append((x,y))
                ways_pixels.append(nodes_pixels)

            self.ways_pixels = ways_pixels[:]
    
    # Takes the image
    def get_rgb(self):
        url = 'https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/{0},{1}/{2}?mapSize={3},{4}&format={5}&key={6}'.format(
                    self.center_lat, self.center_lon, self.zoom, self.width, self.height, IMG_FORMAT, KEY)

        response = getResource(url)
        file = open(os.path.join(self.data_path, RGB) + self.id + '.tiff', 'wb')
        file.write(response.content)
        file.close()

    def get_mask(self):
        if self.ways_pixels:
            mask = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

            for wp, ww in zip(self.ways_pixels, self.ways_width):
                points = np.array(wp)
                cv2.polylines(mask, np.int32([points]), 0, (255,255,255), thickness=ww[0])
                    
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(self.data_path, MASK) + self.id + '.tiff', mask)

    def get_mask_points(self):
        if self.ways_pixels:
            mask = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

            for wp in self.ways_pixels:
                points = np.array(wp)
                for p in points:
                  if p[0] < 1500 and p[0] >= 0 and p[1] < 1500 and p[1] >= 0:
                    mask[p[0],p[1]] = [255,255,255]
                    
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            cv2.imwrite((os.path.join(self.data_path, MASK_POINT) + self.id + '.tiff'), mask)

    # Creates the json file containing each street point in pixel coordinate and its dimension
    def write_json(self):
        img_json = {}
        img_json["ways_width"] = self.ways_width
        img_json["ways_pixels"] = self.ways_pixels

        with open(os.path.join(self.data_path, JSON_DATA) + self.id + '.json', "w") as f:
            dump(img_json, f)

    def test_width(self):
        if self.ways_pixels:
            draw_names = glob(RGB + '*.tiff' )
            draw_ids = []
            for draw_name in draw_names:
                draw_ids.append(draw_name[len(RGB) : -(len(IMG_FORMAT)+1)])

            if self.id not in draw_ids:
                image = cv2.imread(os.path.join(self.data_path, RGB) + self.id + '.tiff')

                output = image.copy()

                for wp, ww in zip(self.ways_pixels, self.ways_width):
                    points = np.array(wp)
                    val = ww[0]
                    color = (0,0,0)
                    if val == WIDTH_A:
                        color = (0,0,255)
                    elif val == WIDTH_B:
                        color = (0,255,0)
                    elif val == WIDTH_C:
                        color = (255,255,255)
                    elif val == WIDTH_D:
                        color = (255,0,0)
                    cv2.polylines(output, np.int32([points]), 0, color, thickness=val)
                        
                cv2.imwrite(os.path.join(self.data_path, MASK) + self.id + '.tiff', output)


class ImgProcess (Process):
    def __init__(self, data_path, centers, p_number, f = ''):
        Process.__init__(self)
        self.data_path = data_path
        self.centers = np.array(centers)
        self.p_number = p_number
        self.f = f

    def run(self):
        split = np.array_split(self.centers, self.p_number)
        processes = [Process(target=self.target, args=(split[i],)) for i in range(self.p_number)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    def target(self, points):
        cnt = 1
        tot = len(points)
        for point in points:
            print(self.f, 'process', os.getpid(), 'image', cnt, '/', tot)
            img = Image(self.data_path, point[0], point[1], ZOOM, WIDTH, HEIGHT)
            img.get_corners()

            # sometimes the api server is busy or you reach the max rate
            time_to_sleep = 0.5
            while(not img.get_ways()):
                time_to_sleep += 0.5
                if time_to_sleep > 5:
                    time_to_sleep = 2
                print('Process', os.getpid(), 'too much data for', cnt, ', sleeping for', time_to_sleep)
                sleep(time_to_sleep)

            # if ways are found for that map section then continue
            if len(img.ways) > 0:
                img.get_rgb()
                img.get_ways_pixels()
                img.get_mask()
                img.write_json()
                # img.test_width()
            else:
                print('Process', os.getpid(), 'image', cnt, 'has no ways')
            
            cnt += 1
            del img
            # sleep(1.5)

        print(self.f, 'process', os.getpid(), 'finished')


def test(url):

    l1 = []
    l2 = []

    xml = getResource(url).text
    soup = bs(xml, 'xml')

    if soup.find(text="Error"):
        print('error')

    node_list = [] # (id,lat,lon)
    nodes_tag = soup.find_all('node')
    for node_tag in nodes_tag:
        id = int(node_tag.attrs['id'])
        lat = float(node_tag.attrs['lat'])
        lon = float(node_tag.attrs['lon'])
        node_list.append((id, lat, lon))

    ref_list = [
        ('highway','motorway'), ('highway','trunk'), ('highway','primary'), ('highway','secondary'),
        ('highway','tertiary'), ('highway','unclassified'),('highway','residential'), 
        
        ('highway','motorway_link'), ('highway','trunk_link'), ('highway','primary_link'),
        ('highway','secondary_link'), ('highway','tertiary_link'),
        
        ('highway','living_street'), ('highway','service'), ('highway','pedestrian'),
        ('highway','bus_guideway'), ('highway','road'), ('highway','busway')]

    width_a = ['motorway']
    width_b = ['trunk','primary']
    width_c = ['secondary', 'tertiary']
    width_d = [
        'motorway_link', 'trunk_link', 'primary_link',
        'secondary_link', 'tertiary_link', 'unclassified',
        'residential','living_street', 'service',
        'pedestrian', 'bus_guideway', 'road', 'busway']
        
    way_list = [] # [(lat,lon)]
    ways_width = []

    ways = soup.find_all('way')
    for way in ways:
        tags_list = [] # (key,value)
        tags = way.find_all('tag')
        for tag in tags:
            k = tag.attrs['k']
            v = tag.attrs['v']
            tags_list.append((k,v))
        
        if len(tags_list) > 0:
            if any(elem in ref_list for elem in tags_list):
                ways_width_tmp = []
                for t in tags_list:
                    if(t[1] in width_a):
                        ways_width_tmp.append(1)
                    elif(t[1] in width_b):
                        ways_width_tmp.append(2)
                    elif(t[1] in width_c):
                        ways_width_tmp.append(3)
                    elif(t[1] in width_d):
                        ways_width_tmp.append(4)
                    
                ways_width.append(ways_width_tmp)
                    
                nodes = [] # (lat,lon)
                node_tags = way.find_all('nd')
                for node_tag in node_tags:
                    id = int(node_tag.attrs['ref'])
                    for n in node_list:
                        if id == n[0]:
                            nodes.append((n[1], n[2]))

                way_list.append(nodes)

    l1 = way_list[:]
    l2 = ways_width[:]

    #for i in l1:
    #  print(len(i))

    #print(l1)

    if len(l1) != len(l2):
        print('ERROR WHILE PARSING RESPONSE')

    if not l1:
        print(xml)

    return l1, l2

def get_corners_test(lat, lon):
    url = 'https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/{0},{1}/{2}?mapSize={3},{4}&key={5}&mapMetadata=1&o={6}'.format(
        lat, lon, 18, 1500, 1500, KEY, DATA_FORMAT)

    html = getResource(url).text
    soup = bs(html, 'html.parser')
    content = loads(soup.text)
    box = content['resourceSets'][0]['resources'][0]['bbox']
    # box -> [S, W, N, E]
    arr = []
    arr.append(box[0])
    arr.append(box[1])
    arr.append(box[2])
    arr.append(box[3])
    return arr

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', help='Path to the file containing the image coordinates.', required=True)
    parser.add_argument('--data', help='Directory to the specified set.', required=True)
    parser.add_argument('--num', help='Number of images for each city.', type=int, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    names = []
    out_files = []
    coo_num = []

    with open(args.config, "r") as file:
        count = 0

        for line in file:
            city = line.split(" #")[0].split(", ")  # Remove comments
            names.append(city)                      # City coordinates
            out_files.append("T" + str(count))      # File name
            coo_num.append(args.num)                # Number of images
            count += 1

    arrs = []

    for i in range(len(coo_num)):
        j = coo_num[i]
        arrs.append(random_walk(float(names[i][0]), float(names[i][1]), float(names[i][2]), float(names[i][3]), j))

    for arr, file in zip(arrs, out_files):
        with open(os.path.join(args.data, TXT_DATA) + file + '.txt', 'w') as f:                    
            for i in arr:
                line = str(i[0]) + ',' + str(i[1]) + ',\n'
                f.write(line)

    p_number = cpu_count()

    to_do = out_files

    for a in to_do:
        print(a)
        centers = []

        with open(os.path.join(args.data, TXT_DATA) + a + '.txt', 'r') as f:
            for line in f:
                el = line.split(',')
                tmp = (float(el[0]), float(el[1]))
                centers.append(tmp)

        proc = ImgProcess(args.data, centers, p_number, a)
        proc.run()
        print(a, 'done\n')

    print('\n\nDONE\n\n')

if __name__ == '__main__':
    main()
