import sys, time, slices
name, k = sys.argv[1], int(sys.argv[2])
H, emp, cyc = slices.slice_family(name)
t0 = time.time(); r = slices.colourable(H, list(cyc), k)
print(f"{name} k={k}: CaDiCaL {'colourable' if r else 'NOT colourable'} ({time.time()-t0:.0f}s) [{len(cyc)} rigid slice triangles]", flush=True)
