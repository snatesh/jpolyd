import re
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Parsing
# -----------------------------------------------------------

def parse_sweep_results(path):
  """
  Parse a sweep_results-style file.

  Returns:
    data: dict of
      data[block_type] = {
        'name': block_name,
        'configs': {
          (dct_keep, jac_deg_keep): [
            {
              'base_step': float,
              'dct_bpp': float,
              'dct_psnr': float,
              'jac_bpp': float,
              'jac_psnr': float,
            }, ...
          ]
        }
      }
  """
  text = Path(path).read_text()

  # Find all block headers
  block_header_re = re.compile(
      r"^============================================================\n"
      r"Block type (\d+) \((.+?)\)",
      re.MULTILINE
  )

  # Truncation config lines
  trunc_re = re.compile(
      r"^=== Truncation config: dct_max_k_keep=(\d+), jac_max_deg_keep=(\d+) ===",
      re.MULTILINE
  )

  # Data lines inside each trunc config
  line_re = re.compile(
      r"^base_step=(\d+(?:\.\d+)?)\s+"
      r"DCT: bpp=([\d.]+)\s+PSNR=([-\d.]+)\s+"
      r"Jac: bpp=([\d.]+)\s+PSNR=([-\d.]+) dB",
      re.MULTILINE
  )

  data = {}

  # Iterate over blocks
  for block_match in block_header_re.finditer(text):
    block_type = int(block_match.group(1))
    block_name = block_match.group(2).strip()
    data.setdefault(block_type, {'name': block_name, 'configs': {}})

    # Find the substring for this block (up to the next "======" or end)
    block_start = block_match.start()
    next_block = block_header_re.search(text, block_match.end())
    block_end = next_block.start() if next_block else len(text)
    block_text = text[block_start:block_end]

    # Inside this block, find all truncation configs
    for trunc_match in trunc_re.finditer(block_text):
      dct_keep = int(trunc_match.group(1))
      jac_deg = int(trunc_match.group(2))

      # Substring for this trunc config (up to next trunc config or block end)
      tc_start = trunc_match.end()
      next_trunc = trunc_re.search(block_text, tc_start)
      tc_end = next_trunc.start() if next_trunc else len(block_text)
      tc_text = block_text[tc_start:tc_end]

      key = (dct_keep, jac_deg)
      cfg_list = data[block_type]['configs'].setdefault(key, [])

      # Parse base_step lines
      for line_match in line_re.finditer(tc_text):
        base_step = float(line_match.group(1))
        dct_bpp   = float(line_match.group(2))
        dct_psnr  = float(line_match.group(3))
        jac_bpp   = float(line_match.group(4))
        jac_psnr  = float(line_match.group(5))
        cfg_list.append({
          'base_step': base_step,
          'dct_bpp':   dct_bpp,
          'dct_psnr':  dct_psnr,
          'jac_bpp':   jac_bpp,
          'jac_psnr':  jac_psnr,
        })

      # Sort by base_step (just for nice plotting)
      cfg_list.sort(key=lambda r: r['base_step'])

  return data


# -----------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------

def list_block_types(data):
  print("Available block types:")
  for bt, info in data.items():
    print(f"  {bt}: {info['name']}")


def list_configs_for_block(data, block_type):
  info = data[block_type]
  print(f"Configs for block {block_type} ({info['name']}):")
  for (dct_keep, jac_deg) in sorted(info['configs'].keys()):
    print(f"  dct_keep={dct_keep}, jac_deg_keep={jac_deg}")


def plot_block_config(data, block_type, dct_keep, jac_deg_keep,
                      show_base_step_labels=False):
  """
  Plot PSNR vs bpp for a single (block_type, dct_keep, jac_deg_keep).
  DCT curve and Jacobi curve on the same axes.
  """
  info = data[block_type]
  block_name = info['name']
  cfg_key = (dct_keep, jac_deg_keep)
  if cfg_key not in info['configs']:
    raise ValueError(f"No data for block {block_type}, config {cfg_key}")

  rows = info['configs'][cfg_key]

  dct_bpp  = [r['dct_bpp']  for r in rows]
  dct_psnr = [r['dct_psnr'] for r in rows]
  jac_bpp  = [r['jac_bpp']  for r in rows]
  jac_psnr = [r['jac_psnr'] for r in rows]
  base_steps = [r['base_step'] for r in rows]

  plt.figure(figsize=(6,4))
  plt.title(f"Block {block_type} ({block_name})\n"
            f"DCT_keep={dct_keep}, Jac_deg<={jac_deg_keep}")
  # DCT
  plt.plot(dct_bpp, dct_psnr, '-o', label='DCT')
  # Jacobi
  plt.plot(jac_bpp, jac_psnr, '-s', label='Jacobi')

  if show_base_step_labels:
    for x, y, bs in zip(dct_bpp, dct_psnr, base_steps):
      plt.text(x, y, f"D{int(bs)}", fontsize=8, color='blue')
    for x, y, bs in zip(jac_bpp, jac_psnr, base_steps):
      plt.text(x, y, f"J{int(bs)}", fontsize=8, color='orange')

  plt.xlabel("bits per pixel (bpp)")
  plt.ylabel("PSNR (dB)")
  plt.grid(True, linestyle='--', alpha=0.4)
  plt.legend()
  plt.tight_layout()


def plot_all_configs_for_block(data, block_type):
  """
  Make a grid of subplots per truncation config for a given block_type.
  Each subplot: PSNR vs bpp (DCT + Jacobi).
  """
  info = data[block_type]
  block_name = info['name']
  configs = sorted(info['configs'].keys())
  n_cfg = len(configs)
  ncols = 3
  nrows = (n_cfg + ncols - 1) // ncols

  fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)
  fig.suptitle(f"Block {block_type} ({block_name}) - all truncation configs",
               fontsize=14)

  for ax, (dct_keep, jac_deg) in zip(axes.flat, configs):
    rows = info['configs'][(dct_keep, jac_deg)]
    dct_bpp  = [r['dct_bpp']  for r in rows]
    dct_psnr = [r['dct_psnr'] for r in rows]
    jac_bpp  = [r['jac_bpp']  for r in rows]
    jac_psnr = [r['jac_psnr'] for r in rows]

    ax.plot(dct_bpp, dct_psnr, '-o', label='DCT')
    ax.plot(jac_bpp, jac_psnr, '-s', label='Jacobi')
    ax.set_title(f"DCT_keep={dct_keep}, Jac_deg<={jac_deg}")
    ax.set_xlabel("bpp")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, linestyle='--', alpha=0.4)

  # Hide any unused subplots
  for ax in axes.flat[len(configs):]:
    ax.axis('off')

  handles, labels = axes[0][0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper right')
  fig.tight_layout(rect=[0, 0, 0.96, 0.95])


# -----------------------------------------------------------
# Main demo
# -----------------------------------------------------------

if __name__ == "__main__":
  # Change this if your file is named differently
  fname = "sweep_results.txt"

  data = parse_sweep_results(fname)

  # See what's in there
  list_block_types(data)
  print()

  # Example: look at block 0 (gradient)
  block_type = 5
  list_configs_for_block(data, block_type)
  print()

  # Example 1: plot a specific truncation config
  # (adjust dct_keep, jac_deg_keep to something that exists for your data)
  dct_keep_example = 20
  jac_deg_example  = 1
  plot_block_config(data, block_type, dct_keep_example, jac_deg_example,
                    show_base_step_labels=True)

  # Example 2: plot all truncation configs for this block as a grid
  plot_all_configs_for_block(data, block_type)

  plt.show()

