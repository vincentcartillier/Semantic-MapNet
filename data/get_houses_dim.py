import os
import json

houses = os.listdir('data/mp3d/')

houses_dim = {}

for house in houses:
        
        file = open('data/mp3d/{}/{}.house'.format(house, house), 'r')
        line = file.readline()
        line = file.readline()

        data = line.split(' ')
        xlo = float(data[23])
        ylo = float(data[24])
        zlo = float(data[25])
        
        xhi = float(data[27])
        yhi = float(data[28])
        zhi = float(data[29])

        center = [(xlo+xhi)/2, (ylo+yhi)/2, (zlo+zhi)/2]
        sizes  = [(xhi-xlo), (yhi-ylo), (zhi-zlo)]

        center = [center[0], center[2], -center[1]]
        sizes = [sizes[0], sizes[2], sizes[1]]


        houses_dim[house] = {'xlo': xlo,
                             'ylo': ylo,
                             'zlo': zlo,
                             'xhi': xhi,
                             'yhi': yhi,
                             'zhi': zhi,
                             'center': center,
                             'sizes': sizes}
        
        line = file.readline()
        while line[0] == 'L':

            data = line.split(' ')
            level = data[2] 

            xlo = float(data[12])
            ylo = float(data[13])
            zlo = float(data[14])
            
            xhi = float(data[16])
            yhi = float(data[17])
            zhi = float(data[18])

            center = [(xlo+xhi)/2, (ylo+yhi)/2, (zlo+zhi)/2]
            sizes  = [(xhi-xlo), (yhi-ylo), (zhi-zlo)]

            center = [center[0], center[2], -center[1]]
            sizes = [sizes[0], sizes[2], sizes[1]]



            houses_dim['_'.join([house, level])] = {'xlo': xlo,
                                                  'ylo': ylo,
                                                  'zlo': zlo,
                                                  'xhi': xhi,
                                                  'yhi': yhi,
                                                  'zhi': zhi,
                                                  'center': center,
                                                  'sizes': sizes}
            line = file.readline()

json.dump(houses_dim, open('data/houses_dim.json', 'w'))
