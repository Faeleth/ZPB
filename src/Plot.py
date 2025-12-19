import numpy as np

class Plot:
    def __init__(self, maxFrames=100):
        self.frames_ = [[] for i in range(maxFrames)]
        self.index_ = 0
        self.counts_ = {
            "Anger": [0, 0.0],
            "Contempt": [0, 0.0],
            "Disgust": [0, 0.0],
            "Fear": [0, 0.0],
            "Happy": [0, 0.0],
            "Neutral": [0, 0.0],
            "Sad": [0, 0.0],
            "Surprise": [0, 0.0]
        }

    def update(self, emotionsOnFrame):
        # usuwamy ramke
        self.removeFrame()
        # zapisujemy ramke
        self.addFrame(emotionsOnFrame)
        #update indexu
        if self.index_ + 1 >= len(self.frames_):
            self.index_ = 0
        else:
            self.index_ += 1
        # wysyłamy dane na wykres
        return self.counts_

    def removeFrame(self):
        currentFrame = self.frames_[self.index_]
        if len(currentFrame) > 0:
            for box in currentFrame:
                if self.counts_[box[0]][1] * self.counts_[box[0]][0] - box[1] < 0:
                    self.counts_[box[0]][1] = 0.0
                elif self.counts_[box[0]][0] - 1 <= 0:
                    self.counts_[box[0]][1] = 0.0
                else:
                    self.counts_[box[0]][1] = (self.counts_[box[0]][1] * self.counts_[box[0]][0] - box[1]) / (self.counts_[box[0]][0] - 1)

                if self.counts_[box[0]][0] - 1 < 0:
                    self.counts_[box[0]][0] = 0
                else:
                    self.counts_[box[0]][0] -= 1

    def addFrame(self, emotionsOnFrame):
        if len(emotionsOnFrame) > 0:
            for box in emotionsOnFrame:
                self.counts_[box[0]][1] = (self.counts_[box[0]][1] * self.counts_[box[0]][0] + box[1]) / (self.counts_[box[0]][0] + 1)
                self.counts_[box[0]][0] += 1
        self.frames_[self.index_] = emotionsOnFrame