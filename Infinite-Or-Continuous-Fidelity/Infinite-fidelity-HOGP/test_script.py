import os
import sys
import subprocess

def run_command(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args)

# %%
domain = "heat"
method = "ifc_ode"
rank = 5
epochs = 5000

domain_name = domain.lower()
save_path = "__res_" + domain_name + "__"

for fold in range(1, 6):
    if method == "ifc_ode":
        run_command([
            "python", "main.py",
            "--config=configs/" + domain_name + "/exp_sf.py",
            "--workdir=" + save_path,
            "--config.training.epochs=" + str(epochs),
            "--config.model.rank=" + str(rank),
            "--config.data.fold=" + str(fold)
        ])
    else:
        print("Error: no such method found..")
