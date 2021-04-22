import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Model
from utils import load_data

def get_survival_rate(voyages):
    return (voyages["damage taken"] < 1).values.mean()

def get_survival_damage(voyages):
    survived = voyages[(voyages["damage taken"] < 1)]
    return survived["damage taken"].values.mean()

def exhaustive_search(reps=1000):

    costs = np.array([40, 20, 45, 1, 10, 35, 15])

    model = Model()
    data = load_data()
    model.fit(data, verbose=False)

    intervention_space = dict(
        shark_repellant=(False, True),
        woodworkers=(False, True),
        merpeople_tribute=(False, True),
        extra_oars=range(20 + 1),
        extra_cannons=range(3 + 1),
        arm_crows_nest=(False, True),
        foam_swords=(False, True),
    )

    best_results = {}
    for values in tqdm(
        itertools.product(*intervention_space.values()),
        total=np.prod([len(v) for v in intervention_space.values()]),
    ):
        interventions = {
            k: v for k, v in zip(intervention_space, values)
        }

        values = np.array(values)
        gold = 100 - np.dot(costs, values)

        voyages = model.make_voyage(**interventions, reps=reps)
        survival_rate = get_survival_rate(voyages)

        if survival_rate > 0.95:
            # Increase accuracy
            voyages = model.make_voyage(**interventions, reps=100*reps)
            survival_rate = get_survival_rate(voyages)
            if (
                gold not in best_results
                or survival_rate > best_results[gold]["survival_rate"]
            ):
                best_results[gold] = dict(
                    survival_rate=survival_rate,
                    interventions=interventions,
                    survival_damage=get_survival_damage(voyages),
                )
                
                if gold >= 0:
                    print("\rGold remaining", gold)
                    print("Survival fraction", survival_rate)

    return best_results

def main():
    # Exhaustive search
    filename = "best_results.pickle"
    if not os.path.isfile(filename):
        best_results = exhaustive_search()
        with open(filename, "wb") as f:
            pickle.dump(best_results, f)

    with open(filename, "rb") as f:
        best_results = pickle.load(f)

    for k, v in best_results.items():
        if "frac_survived" in v:
            v["survival_rate"] = v["frac_survived"]
         

    fig, ax = plt.subplots()
    gold = np.array(list(best_results.keys()))
    survival_rate = [v["survival_rate"] for v in best_results.values()]
    ax.plot(gold, survival_rate, '.')
    ax.set_xlabel("Gold remaining")
    ax.set_ylabel("Survival rate")

    fig.tight_layout()
    fig.savefig("gold_vs_survival.pdf", bbox_inches="tight")

    i_best = np.argmax(survival_rate - 1000 * (gold < 0))
    gold_best = gold[i_best]
    print(f"Best result is achieved by spending {gold_best} gold on")
    for k, v in best_results[gold_best]["interventions"].items():
        if v==1:
            print(k)
        elif v>1:
            print(v, k)

    # Recalculate best
    model = Model()
    data = load_data()
    model.fit(data)

    voyages = model.make_voyage(
        merpeople_tribute=True,
        woodworkers=True,
        extra_oars=20,  # 19 is probably from randomness
        extra_cannons=1,
        reps=1000000,
    )
    print("Frequecy of survial", get_survival_rate(voyages))
    print("Average damage sustained on survivors", get_survival_damage(voyages))
    print("Gold left", 5)

    fig, ax = plt.subplots()
    ax.hist(voyages["damage taken"], density=True, bins=500)
    ax.set_xlabel("Damage received from 10 trips")
    ax.set_ylabel("Frequency")

    plt.show()
    fig.tight_layout()
    fig.savefig("damage_per_trip.pdf", bbox_inches="tight")
    

if __name__=="__main__":
    main()
