import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, x_cont, y_cont, saved_name):
    fig = plt.figure(figsize=(6, 5), tight_layout=True)
    plt.xlabel(x_cont, size=18)
    plt.ylabel(y_cont, size=18)
    plt.xticks(size=16)
    plt.yticks(size=16)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['top'].set_linewidth(1.3)
    ax.spines['right'].set_linewidth(1.3)
    plt.grid(linestyle='-.')
    plt.plot(x, y, 'rD-', color='#ffc20e', linewidth=2,
             linestyle='dashed', markersize=8)
    fig.savefig('/Users/zhulf/Pictures/{}.pdf'.format(saved_name), dpi=1000)
    plt.show()
    plt.close(fig)


iters = [1, 2, 3, 4, 5]
iters_R_error = [1.8551, 1.4656, 1.3893, 1.3608, 1.3489]
iters_t_error = [0.0179, 0.0145, 0.0144, 0.0144, 0.0143]
iters_x_cont = ['Iter', 'Iter']
iters_y_cont = ['Error (R) (degree)', 'Error (t)']
iters_saved_name = ['iter-r', 'iter-t']
plot(np.array(iters).astype(dtype=np.str), iters_R_error, iters_x_cont[0], iters_y_cont[0], iters_saved_name[0])
plot(np.array(iters).astype(dtype=np.str), iters_t_error, iters_x_cont[1], iters_y_cont[1], iters_saved_name[1])


topn1 = [717, 560, 448, 336, 224]
topn1_R_error = [1.5161, 1.4697, 1.4656, 1.5931, 1.9068]
topn1_t_error = [0.0152, 0.0146, 0.0145, 0.0159, 0.0186]
topn1_x_cont = ['top-N1', 'top-N1']
topn1_y_cont = ['Error (R) (degree)', 'Error (t)']
topn1_saved_name = ['n1-r', 'n1-t']
plot(topn1, topn1_R_error, topn1_x_cont[0], topn1_y_cont[0], topn1_saved_name[0])
plot(topn1, topn1_t_error, topn1_x_cont[1], topn1_y_cont[1], topn1_saved_name[1])


topm1 = [717, 560, 448]
topm1_R_error = [1.4656, 1.6011, 1.8051]
topm1_t_error = [0.0145, 0.0156, 0.0175]
topm1_x_cont = ['top-M1', 'top-M1']
topm1_y_cont = ['Error (R) (degree)', 'Error (t)']
topm1_saved_name = ['m1-r', 'm1-t']
plot(topm1, topm1_R_error, topm1_x_cont[0], topm1_y_cont[0], topm1_saved_name[0])
plot(topm1, topm1_t_error, topm1_x_cont[1], topm1_y_cont[1], topm1_saved_name[1])


topprob = [1, 0.8, 0.6, 0.4, 0.2]
topprob_R_error = [2.2960, 1.6823, 1.5329, 1.4656, 1.5278]
topprob_t_error = [0.0234, 0.0166, 0.0153, 0.0145, 0.0151]
topprob_x_cont = ['top-prob', 'top-prob']
topprob_y_cont = ['Error (R) (degree)', 'Error (t)']
topprob_saved_name = ['prob-r', 'prob-t']
plot(topprob, topprob_R_error, topprob_x_cont[0], topprob_y_cont[0], topprob_saved_name[0])
plot(topprob, topprob_t_error, topprob_x_cont[1], topprob_y_cont[1], topprob_saved_name[1])
