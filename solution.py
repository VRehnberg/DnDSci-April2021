import os
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Model
from utils import load_data

def exhaustive_search(reps=1000):

    costs = np.array([40, 20, 45, 1, 10, 35, 15])

    model = Model()
    data = load_data()
    model.fit(data, verbose=False)

    intervention_space = dict(
        shark_repellant=(False, True),
        woodworkers=(False, True),
        merpeople_tribute=(False, True),
        extra_oars=range(20),
        extra_cannons=range(3),
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

        damage_taken = model.make_voyage(**interventions, reps=reps)
        frac_survived = (damage_taken < 1).sum() / len(damage_taken)

        if frac_survived > 0.005:
            # Increase accuracy
            damage_taken = model.make_voyage(**interventions, reps=100*reps)
            frac_survived = (damage_taken < 1).sum() / len(damage_taken)
            if (
                gold not in best_results
                or frac_survived < best_results[gold]["frac_survived"]
            ):
                best_results[gold] = dict(
                    frac_survived=frac_survived,
                    interventions=interventions,
                    survival_damage=damage_taken[damage_taken < 1].mean(),
                )
                
                if gold >= 0:
                    print("\rGold remaining", gold)
                    print("Survival fraction", frac_survived)

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

    for gold, res in best_results.items():
        if gold >= 0:
            print(gold, res["frac_survived"], res["survival_damage"])
            print(res["interventions"])


    # Recalculate best
    model = Model()
    data = load_data()
    model.fit(data)

    damage_taken = model.make_voyage(
        shark_repellant=True,
        merpeople_tribute=True,
        extra_oars=15,
        reps=500000,
    )
    is_survival = (damage_taken<1)
    print("Frequecy of survial", is_survival.sum() / is_survival.size) 
    print("Average damage sustained on survivors", damage_taken[is_survival].mean())
    print("Gold left", 0)

    fig, ax = plt.subplots()
    ax.hist(damage_taken, density=True, bins=500)
    ax.set_xlabel("Damage received from 10 trips")
    ax.set_ylabel("Frequency")
    plt.show()
    fig.tight_layout()
    fig.savefig("damage_10_trips.pdf", bbox_inches="tight")
    

if __name__=="__main__":
    main()
