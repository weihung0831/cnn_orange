import matplotlib.pyplot as plt
import pretty_errors
import seaborn as sns


def training_model_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.title('Training and Validation Accuracy')
    plt.grid(linestyle=':')
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])

    plt.subplot(212)
    plt.title('Training and Validation Loss')
    plt.grid(linestyle=':')
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylabel('CategoricalCrossentropy')
    plt.ylim([0, max(plt.ylim())])
    plt.legend(loc='upper right')
    plt.savefig("./model/model2/training_result.png")
    plt.show()


def matrix_plot(matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predict')
    plt.ylabel('True')
    plt.savefig('./model/model2/model_matrix.png')
    plt.show()

    return matrix
