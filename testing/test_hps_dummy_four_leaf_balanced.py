from jhps import load_library, run_four_leaf_balanced_test


def main():
  results = []
  tol = 1.0e-10
  lib = load_library()

  for D in range(1, 6):
    res = run_four_leaf_balanced_test(
      D,
      kf=2,
      vol_dim=D + 6,
      seed=3000 + D,
      alpha=1.0,
      beta=1.0,
      verbose=True,
      lib=lib,
    )
    results.append(res)

    assert res.root_robin_residual_inf < tol, res
    assert res.interface_flux_residual_inf < tol, res
    assert res.parent_consistency_residual_inf < tol, res
    assert res.monolithic_trace_residual_inf < tol, res
    assert res.root_nb > 0, res
    assert res.interface_nb > 0, res

  print("\nall D=1..5 four-leaf balanced dummy HPS tests passed")
  for res in results:
    print(res)


if __name__ == "__main__":
  main()
