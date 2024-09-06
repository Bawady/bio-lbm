import test
import os
import time
from utils.dump_util import *
from datetime import datetime
from src.MultiLBM import MultiLBM


if __name__ == '__main__':
    # Set desired test here
    sim_type = "Poiseuille"
    sim_subtype = "tmp"
    runs = 5000
    dump_period = 500

    store_at = "test/sim_out/" + sim_type + "/" + sim_subtype

    if not store_at.endswith("/"):
        store_at += "/"
    store_at += "{date:%d_%m_%H_%M_%S}".format(date=datetime.now())
    chkpt_dir = store_at + "/chkpts"

    if not os.path.exists(store_at):
        os.makedirs(store_at)
    sim = test.factory.create(sim_type)
    sim.serialize(chkpt_dir + "/setup")
#    sim = MultiLBM.deserialize("test/sim_out/BacterialADR/reproduce/16_08_19_11_27/chkpts/results")

    sim.run(runs, dump_period, store_at)
    print(f"Simulated time: {time.strftime('  %H:%M:%S', time.gmtime(mag(sim.time(unit='s'))))}")

    sim.serialize(chkpt_dir + "/results")

    sim.post_sim(show=True, store_at=store_at)
    print("Simulation done")
