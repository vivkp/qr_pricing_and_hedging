from convergence.mc_convergence import run_mc_convergence
from convergence.fd_convergence import run_fd_convergence


def main():
    mc_png, mc_csv = run_mc_convergence()
    fd_png, fd_csv = run_fd_convergence()

    print("\nConvergence experiments complete.")
    print(f"Monte Carlo graph      : {mc_png}")
    print(f"Monte Carlo data       : {mc_csv}")
    print(f"Finite Difference graph: {fd_png}")
    print(f"Finite Difference data : {fd_csv}")


if __name__ == "__main__":
    main()
