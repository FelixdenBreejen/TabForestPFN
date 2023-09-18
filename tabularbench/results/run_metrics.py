from dataclasses import dataclass, field


@dataclass
class RunMetrics():
    scores_train: list[float] = field(default_factory=list)
    scores_val: list[float] = field(default_factory=list)
    scores_test: list[float] = field(default_factory=list)
    losses_train: list[float] = field(default_factory=list)
    losses_val: list[float] = field(default_factory=list)
    losses_test: list[float] = field(default_factory=list)


    def append(self, score_train: float, score_val: float, score_test: float, loss_train: float, loss_val: float, loss_test: float):
        self.scores_train.append(score_train)
        self.scores_val.append(score_val)
        self.scores_test.append(score_test)
        self.losses_train.append(loss_train)
        self.losses_val.append(loss_val)
        self.losses_test.append(loss_test)


    def __len__(self):
        return len(self.scores_train)