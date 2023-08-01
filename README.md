# AutoCausality_withRealcause
- FLAML_Modify : modification for FLAML
- AutoCausality_modify : modification for AutoCausality



# Setup Steps:
1. install FLAML
2. unzip the autocausality_20230801.tar.gz to python lib path (usually ~/.local/lib/python3.X/site-package/autocausality )
3. for FLAML>=1.0.14, put those py files under FLAML_modify to ~/.local/lib/python3.X/site-package/flaml/automl/
4. put those files under AutoCausality_modify to ~/.local/lib/python3.X/site-package/autocausality
5. git clone https://github.com/bradyneal/realcause.git
6. put monkey_patch.py under the git folder realcause
7. put autocausality_AutoSuper.py under the git folder realcause
8. under realcasuse directory, mkdir Super_NewData ( for storing run results)
