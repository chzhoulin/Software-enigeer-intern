def drstrdrv(fYawAcc, gSpeed):
    fYawAcc = fYawAcc - 320;

    if(fYawAcc > -6) and (fYawAcc < 6):
        fYawAcc = fYawAcc * 0.1 ;

    gfYawAccOld = fYawAcc;
    fYawAcc = fYawAcc + 0.5*(fYawAcc - gfYawAccOld);

    mfLaneYawacc = fYawAcc*(-0.2)
    if(gSpeed > 0) and (gSpeed < 200):
        mfLaneYawacc = mfLaneYawacc/(gSpeed*gSpeed/100.0);

    return mfLaneYawacc
