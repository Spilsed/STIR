import matplotlib.pyplot as plt

with open("WCLIP1698540141304.1511_val.txt", "r") as f:
    text = f.read()
    points = text[1:-1].split(", ")

    for point in range(len(points)):
        points[point] = float(points[point])


    plt.title("R-CNN w/ CLIP Validation")
    plt.plot(points)
    plt.show()
