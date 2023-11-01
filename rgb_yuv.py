import sys

import numpy as np


a=input("press 1 if you'd like to convert RGB to YUV and way round; 2 to convert directly YUV to RGB;3 to convert directly YUV to RGB; press any other number to exit")
a=int(a)
####conversion functions
def rgb_to_yuv_back(r, g, b):
    Y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    U = -0.148 * r - 0.291 * g + 0.439 * b + 128
    V = 0.439 * r - 0.368 * g - 0.071 * b + 128
    Y -= 16
    U -= 128
    V -= 128
    R2 = 1.164 * Y + 1.596 * V
    G2= 1.164 * Y - 0.392 * U - 0.813 * V
    B2= 1.164 * Y + 2.017 * U
    y = [R2, G2, B2]
    print("the correspondent new RGB array is: ")
    print(y)
    return y
def yuv_to_rgb(Y,U,V):
    Y -= 16
    U -= 128
    V -= 128
    R2 = 1.164 * Y + 1.596 * V
    G2 = 1.164 * Y - 0.392 * U - 0.813 * V
    B2 = 1.164 * Y + 2.017 * U
    z=[R2,G2,B2]
    print("the correspondent RGB array is: ")
    print(z)
    return z
def rgb_to_yuv(r, g, b):
    Y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    U = -0.148 * r - 0.291 * g + 0.439 * b + 128
    V = 0.439 * r - 0.368 * g - 0.071 * b + 128
    h=[Y,U,V]
    print("the correspondent YUV array is: ")
    print(h)
    return h

if(a==1):
     r=input("insert the value for R")
     g=input("insert the value for G")
     b=input("insert the value for B")
     r=float(r)
     g=float(g)
     b=float(b)
     rgb_to_yuv_back(r,g,b)
elif(a==2):
    y2 = input("insert the value for Y")
    u2 = input("insert the value for U")
    v2= input("insert the value for V")
    y2=float(y2)
    u2=float(u2)
    v2=float(v2)
    yuv_to_rgb(y2,u2,v2)
elif(a==3):
     r=input("insert the value for R")
     g=input("insert the value for G")
     b=input("insert the value for B")
     r=float(r)
     g=float(g)
     b=float(b)
     rgb_to_yuv(r,g,b)
else:
    exit()


####REDUCING IMAGE QUALITY

import subprocess as sp



def reduce_quality(input,output,width,height):
    resize=[

        'ffmpeg',
        '-i', input,  # input
        '-vf', f'scale={width}:{height}',  # filter
        output #output
    ]
    y=sp.run(resize)
    if sp.run(resize).returncode == 0:
        print("FFmpeg Script Ran Successfully")

    return y


x=input("let's resize an image, press enter to continue")
reduce_quality('Practica1.jpg','/mnt/c/users/danie/PycharmProjects/pythonProject/ex2output.jpg',500,600)


##EX3
def serpentine(input):
    #here i open the file, transforming it into a 8x8 matrix

    with open(input,"rb") as image:
        x=image.read()
        x=bytearray(x)
        sys.set_int_max_str_digits(0)
        x=int.from_bytes(x,"big")
        x=np.array(x)

        x=np.resize(x,(8,8))

        lst=[]     ##list to store the values

        i=0
        j=0    ##inizialize the indeces for the matrix
        byte = x[i][j]        ##store the first value
        lst.append(byte)
        ##starting the actual serpentine
        while len(lst)<=63:
        ##getting the first half of the matrix(the calculation takes into account that
        ##the first position of an array is setted as 0 and not a 1, so the last position
        ##of the matrix will be [7,7] not [8,8]
            while i!=7:

                if j%2==0 and i==0:
                    byte=x[i][j+1]
                    lst.append(byte)
                    j+=1
                elif j%2 and i==0:
                    k=1
                    while k<=j:
                        byte=x[i+k][j-k]
                        lst.append(byte)
                        k+=1
                    

                    i=i+(k-1)
                    j=j-(k-1)
                elif i%2 and j==0:
                    byte=x[i+1][j]
                    lst.append(byte)
                    i+=1
                elif i%2==0 and j==0:
                    k = 1
                    while k <=i:
                        byte = x[i - k][j + k]
                        lst.append(byte)
                        k += 1
                    i=i-(k-1)
                    j=j+(k-1)

            while j!=7 or i!=7:             ###here i pass the "second half" of the image
                if j%2==0 and i==7:
                    byte=x[i][j+1]
                    lst.append(byte)
                    j+=1
                elif j%2 and i==7:
                    k=1
                    while k<=(7-j):
                        byte=x[i-k][j+k]
                        lst.append(byte)
                        k+=1
                    i=i-(k-1)
                    j=j+(k-1)
                elif i%2 and j==7:
                    byte = x[i+1][j]
                    lst.append(byte)
                    i += 1
                elif i%2==0 and j==7:
                    k=1
                    while k<=(7-i):
                        byte=x[i+k][j-k]
                        lst.append(byte)
                        k+=1
                    i=i+(k-1)
                    j=j-(k-1)

    return lst

x=input("let's run the serpentine algorithm,press enter to continue")
serpentine('Practica1.jpg')



##EX4

def convert_to_bw(input,output):
    bw=[
        'ffmpeg',
        '-i', input,
        '-vf', f'hue = s = 0',
        output
    ]
    y=sp.run(bw)


    if sp.run(bw).returncode == 0:
        print("FFmpeg Script Ran Successfully")

    return y
x=input("let's make an image in black and white, press enter to continue")
convert_to_bw('Practica1.jpg','/mnt/c/users/danie/PycharmProjects/pythonProject/ex4output.jpg')

##EX5


def encode(message):
    encoded_message = ""
    i = 0

    while (i <= len(message) - 1):
        count = 1
        ch = message[i]
        j = i
        while (j < len(message) - 1):
            if (message[j] == message[j + 1]):
                count = count + 1
                j = j + 1
            else:
                break
        encoded_message = encoded_message + str(count) + ch
        i = j + 1
    return encoded_message
x=input("Time to encode, please insert a string you want to encode:")
z=encode(x)
print(z)

##EX6

class DCT:
    def __init__(self, input):
        self.input = input
    from numpy import r_
    f=input
    n = 8  # This will be the window in which we perform our DCT
    sumd = 0  # INI value

    # Create some blank matrices to store our data

    dctmatrix = np.zeros(np.shape(f))  # Create a DCT matrix in which to plug our values :)
    f = f.astype(np.int16)  # Convert so we can subtract 128 from each pixel
    f = f - 128  # As said above
    f2 = np.zeros(np.shape(f))  # This will be where the compressed image goes

    def cosp(i, j, n):  # This is the funky cos function inside the DCT
        output = 0
        output = np.cos(((2 * i) + 1) * j * np.pi / (2 * n))
        return output

    def convolveDCT(f, n, u, v, a, b):  # This convolve function compute DCT for nxn @ axb location
        sumd = 0  # INI value
        for x in np.r_[0:n]:
            for y in np.r_[0:n]:
                u = u % n
                v = v % n
                sumd += f[x + a, y + b] * DCT.cosp(x, u, n) * DCT.cosp(y, v, n)
        # Now, need to perform the functions outside of the sum values
        if u == 0:
            sumd *= 1 / np.sqrt(2)
        else:
            sumd *= 1
        if v == 0:
            sumd *= 1 / np.sqrt(2)
        else:
            sumd *= 1
        sumd *= 1 / np.sqrt(2 * n)

        return sumd

    for a in r_[0:np.shape(f)[0]:n]:
        for b in r_[0:np.shape(f)[1]:n]:
            # Below, compute the DCT for a given uxv location in the DCT Matrix
            for u in r_[a:a + n]:
                for v in r_[b:b + n]:
                    dctmatrix[u, v] = convolveDCT(f, n, u, v, a, b)

    def convolveIDCT(dctmatrix, n, x, y, a, b):  # This convolve function compute IDCTfor nxn @ axb location
        sumd = 0  # INI value
        for u in np.r_[0:n]:
            for v in np.r_[0:n]:
                val1 = 1
                val2 = 1
                x = x % n
                y = y % n
                if u == 0: val1 = 1 / np.sqrt(2)
                if v == 0: val2 = 1 / np.sqrt(2)
                sumd += dctmatrix[u + a, v + b] * val1 * val2 * DCT.cosp(x, u, n) * DCT.cosp(y, v, n)
        # Now, need to perform the functions outside of the sum values
        sumd *= 2 / n
        return sumd

    # And re run it to get our new compressed image! :)
    # First we need to take into account our multiple nxn windows that jump across the image
    for a in r_[0:np.shape(dctmatrix)[0]:n]:
        for b in r_[0:np.shape(dctmatrix)[1]:n]:
            # Below, compute the IDCT for a given x,y location in the Image Matrix
            for x in r_[a:a + n]:
                for y in r_[b:b + n]:
                    f2[x, y] = convolveIDCT(dctmatrix, n, x, y, a, b)

    f2 = f2 + 128  # Scale our values back to 0-255 so we can see it!

