import os 


path = '../log/spoon2/cut/'
imgs = os.listdir(path)
imgs = [img[:-10] for img in imgs if (img[:2]=='LC' and img[-9:-4]=='color')]

# print(imgs)

subfigures = ['Color', 'GT', 'Spectral', 'SNet', 'UNet']
names = ['color', 'mask', 'spectral', 'my', 'unet']
for (subfigure, name) in zip(subfigures, names):
    print('    \subfigure[' + subfigure + ']{')
    print('        \\begin{minipage}[b]{0.15\\linewidth}')

    for img in imgs:
        print('            \\includegraphics[width=1\\linewidth]', end='')
        print('{' + path + img + '_' +  name + '.jpg}\\vspace{4pt}')
        print('            \\includegraphics[width=1\\linewidth]', end='')
        print('{' + path + 'tmp_cut_' + img + '_' +  name + '.jpg}\\vspace{4pt}')


    print('        \end{minipage}')
    print('    }')


'''
    \subfigure[Color]{
        \begin{minipage}[b]{0.15\linewidth}
            \includegraphics[width=1\linewidth]{../log/cut/LC80150312014226LGN00_08297_color.jpg}\vspace{4pt}
            \includegraphics[width=1\linewidth]{../log/cut/tmp_cut_LC80150312014226LGN00_08297_color.jpg}\vspace{4pt}
            \includegraphics[width=1\linewidth]{../log/cut/LC80650182013237LGN00_10854_color.jpg}\vspace{4pt}
            \includegraphics[width=1\linewidth]{../log/cut/tmp_cut_LC80650182013237LGN00_10854_color.jpg}\vspace{4pt}
            \includegraphics[width=1\linewidth]{../log/cut/LC81620432014072LGN00_16101_color.jpg}\vspace{4pt}
            \includegraphics[width=1\linewidth]{../log/cut/tmp_cut_LC81620432014072LGN00_16101_color.jpg}
        \end{minipage}
    }
'''
   
