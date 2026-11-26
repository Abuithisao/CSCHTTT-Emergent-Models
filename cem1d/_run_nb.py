import sys
import nbformat
from nbclient import NotebookClient

src, dst = sys.argv[1], sys.argv[2]
smoke = len(sys.argv) > 3 and sys.argv[3] == "smoke"

nb = nbformat.read(src, as_version=4)

if smoke:
    c0 = nb.cells[0].source
    c0 = c0.replace("TARGET_GEN = 70", "TARGET_GEN = 2")
    c0 = c0.replace("train_rollouts=100,", "train_rollouts=2,")
    c0 = c0.replace("eval_rollouts=200,", "eval_rollouts=2,")
    c0 = c0.replace("LIVE_PLOT = True", "LIVE_PLOT = False")
    nb.cells[0].source = c0

client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": "."}})
try:
    client.execute()
finally:
    nbformat.write(nb, dst)

for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            print(f"--- cell {i} [{out.name}] ---")
            print(out.text[-3000:])
        elif out.output_type == "error":
            print(f"--- cell {i} [ERROR] ---")
            print("\n".join(out.traceback)[-4000:])
print("DONE")
