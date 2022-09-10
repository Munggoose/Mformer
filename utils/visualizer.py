import numpy as np
import matplotlib.pyplot as plt



def graph_visual(board,x, y,label=None):
    
    board.plot(x,y,label=label)
    board.legend()

    return board




if __name__=='__main__':

    x = [1,2,3,4]
    y = [1,2,3,4]
    fig, ax = plt.subplots(2, 1, figsize=(8,6))
    ax[0] = graph_visual(ax[0],x,y,'check')

    plt.show()
