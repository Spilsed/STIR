import matplotlib.pyplot as plt

with open("1693836841639.558.txt", "r") as f:
    text = f.read()
    points = text[1:-1].split(", ")

    for point in range(len(points)):
        points[point] = float(points[point])


    plt.title("R-CNN w/ CLIP")
    plt.plot(points)
    plt.show()
