def Hchain_geometry(choice,n_hydrogens,R,r_H2=0.7):

    string_geo  = "0 1\n"
    if choice == "linear":
        for d in range(n_hydrogens//2):
            string_geo += "H 0. 0. {}\n".format(- (R/2. + d*R))
            string_geo += "H 0. 0. {}\n".format(+ (R/2. + d*R))
    elif choice == "linear_broken":
        delta_R = np.random.rand(n_hydrogens)/100. # array of random values between [0,1/100)
        for d in range(n_hydrogens//2):
            string_geo += "H 0. 0. {}\n".format(- (R/2. + d*R) + delta_R[d])
        for d in range(n_hydrogens//2-1):
            string_geo += "H 0. 0. {}\n".format(+ (R/2. + d*R) + delta_R[n_hydrogens//2+d])
        string_geo += "H 0. 0. {}\n".format(+ (R/2. + (n_hydrogens//2-1)*R) + delta_R[n_hydrogens-1])
    elif choice == "dimer_horizontal":
        for d in range(n_hydrogens//4):
            string_geo += "H 0. 0. {}\n".format(- (R/2. + d*R + r_H2*d))
            string_geo += "H 0. 0. {}\n".format(+ (R/2. + d*R + r_H2*d))
            string_geo += "H 0. 0. {}\n".format(- (R/2. + d*R + r_H2*(d+1)))
            string_geo += "H 0. 0. {}\n".format(+ (R/2. + d*R + r_H2*(d+1)))
    elif choice == "dimer_vertical":
        for d in range(n_hydrogens//4):
            string_geo += "H 0. {} {}\n".format(-r_H2/2., - (R/2. + d*R))
            string_geo += "H 0. {} {}\n".format(+r_H2/2., - (R/2. + d*R))
            string_geo += "H 0. {} {}\n".format(-r_H2/2., + (R/2. + d*R))
            string_geo += "H 0. {} {}\n".format(+r_H2/2., + (R/2. + d*R))
    string_geo += "symmetry c1\n"
    string_geo += "nocom\n"
    string_geo += "noreorient\n"

    return string_geo
