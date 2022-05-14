import os
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        # possibly handle other errno cases here, otherwise finally:
        else:
            raise

def runos(cmd,cluster=False,header=None,jobname=None,logpath=None,mem="100000",partition="short,long"):
    default_header = """#!/bin/bash
    #SBATCH --job-name=$jobname
    #SBATCH -n 1
    #SBATCH --mem $mem
    #ulimit -l unlimited
    #ulimit -s unlimited
    #ulimit -a
    #SBATCH -o $logpath/$jobname.out
    #SBATCH -e $logpath/$jobname.err
    #SBATCH --partition=$partition
    #SBATCH --export=ALL
    """
    
    if cluster == True:
        
        if jobname == None:
            import datetime
            dtstr = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            jobname = cmd.split(" ")[0].split("/")[-1]+f"_{dtstr}"
        
        if logpath == None:
            logpath = "/fefs/aswg/workspace/mireia.nievas/jobs/logs"
            mkdir_p(logpath)
        
        if header == None:
            header = str(default_header)
            header = header.replace("$jobname",jobname)
            header = header.replace("$logpath",logpath)
            header = header.replace("$mem",mem)
            header = header.replace("$partition",partition)
            
        content = f"{header}\n{cmd}\n"
        
        job_script=f"/fefs/aswg/workspace/mireia.nievas/jobs/{jobname}.sh"
        with open(job_script,"w+") as f:
            f.write(content)
        #print(f"Running job {job_script}")
        print(f"Running job {job_script}")
        print(os.popen(f"ls {job_script}").read())
        print(os.popen(f"cat {job_script}").read())
        print(os.popen(f"sbatch {job_script}").read())
    else:
        print(os.popen(cmd).read())
        
def squeue():
    print("Jobs owned by me")
    runos("squeue | grep mireia")