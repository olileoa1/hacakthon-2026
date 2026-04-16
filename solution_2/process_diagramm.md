# Call Process Flow

This document contains the machine-readable flowchart mapping the two-stage decision tree and interaction flow for the internet services chatbot.

## Diagram

```mermaid
%%{init: {'theme': 'neutral'}}%%
graph TD
    classDef bot fill:#fee,stroke:#f00,stroke-width:2px;
    classDef client fill:#eee,stroke:#000,stroke-width:2px;
    classDef data fill:#ccc,stroke:#999,stroke-width:1px,rx:5,ry:5;
    
    %% LEGEND
    note right of Start: VISUAL LEGEND (Modelled visually with styles)<br/>[Red Border: Bot Action]<br/>[Black Border: Client Action]<br/>[Gray Terminal Node: Data to Save item]

    Start((Process Start)) --> G(Greeting: Identification, ask how it can help)
    G --> C_Int(Client: I want internet at home)
    C_Int --> Q_Cl(Bot decision: Are you already a client?)
    
    %% Level 1 Flows
    subgraph "Level 1: Basic data collection"
        Q_Cl --(Yes)--> D_Cl_Id(Ask for Client Nr / Collect Name&Surname)
        D_Cl_Id --> C_Cl_Id(Provide Name/Nr details)
        C_Cl_Id --> Converge1
        C_Cl_Id --(Capture Data)--> D_Cl_Details_Data(Name & Surname<br/>Nr of subscriber)
        
        Q_Cl --(No)--> Converge1
        
        Converge1 --> D_Age(Ask Age)
        D_Age --> C_Age(Provide Age)
        C_Age --> D_Age_Data( <26 then eligible for Cube Xcite)
        C_Age --> D_Users(Ask for nr of users)
        D_Users --> C_Users(Provide Nr users)
        C_Users --(Capture Data)--> D_Users_Data(Nr Users)
        C_Users --> Prepare_Data(Prepare address logic and calculate Min Speed)
        
        Prepare_Data --> D_Addr(Ask for Address)
        D_Addr --> C_Addr(Provide Address)
        C_Addr --(Capture Data)--> D_Addr_Data(Address)
        C_Addr --> Prepare_Data
        
        %% Oval logic simplified to data preparation chain
        Prepare_Data --> Load_Conceptual_State(All address details & Calculated MinSpeed data conceptual node)
        Load_Conceptual_State --(Capture Data)--> D_Oval_Data(Conceptual State: Min speed 50mb x Users, address data loading)
        Load_Conceptual_State --> Check_Data_State(All required data present?)
        Check_Data_State --(No, Addr missing)--> Prepare_Data
        Check_Data_State --(Yes)--> S_Chk(Bot: Speed check internal logic)
        
        S_Chk --> Dec_Prod(Internal: Decide product path)
        Dec_Prod --(High speed/coverage)--> Out_F(Recommend Fix package)
        Dec_Prod --(Low speed/no coverage)--> Out_C(Recommend Cube package)
        
        Out_F --> L1_End
        Out_C --> L1_End
        L1_End((Level 1 Exit))
    end
    
    %% Transition
    L1_End --> L2_Start((Level 2 Start))
    
    %% Level 2 Flow
    subgraph "Level 2: Negotiation"
        L2_Start --> Rec(Recommendation: present offer)
        Rec --> Q_Rec(Client decision: ACCEPT / REJECT)
        Q_Rec --(REJECT)--> Rec
        Q_Rec --(ACCEPT)--> D_FinalOff_Capture(Final offer decided)
        D_FinalOff_Capture --(Data Save)--> D_FinalOff_Data(Final offer)
        D_FinalOff_Capture --> TV_Up(Bot: TV UPSELLING pitch)
        
        TV_Up --> Q_TV(Client decision: ACCEPT / REJECT)
        Q_TV --(REJECT)--> TV_Up
        Q_TV --(ACCEPT)--> D_FinalTVOff_Capture(Final TV offer decided)
        D_FinalTVOff_Capture --(Data Save)--> D_FinalTVOff_Data(Final TV offer)
        D_FinalTVOff_Capture --> Dec_ProdL2(Internal Bot check: Check initial package type)
        
        %% Branches corrected to sensible ending logic
        Dec_ProdL2 --(CUBE product)--> Move_to_Agent(Bot closing node: thank and move to agent)
        Dec_ProdL2 --(FIX product)--> FIX_Prop(Propose appointment with tech)
        
        FIX_Prop --> Q_Appt(Client decision: ACCEPT / REJECT)
        Q_Appt --(REJECT)--> FIX_Prop
        Q_Appt --(ACCEPT)--> D_FinalAppt_Capture(Final appointment scheduled)
        D_FinalAppt_Capture --(Data Save)--> D_FinalAppt_Data(Final appointment)
        D_FinalAppt_Capture --> Move_to_Agent
        
        Move_to_Agent --> Call_Complete((Call complete / Agent handoff))
    end
    
    %% Style Assignment (applying visually to meet legends)
    class G,D_Cl_Id,D_Age,D_Users,Prepare_Data,D_Addr,Load_Conceptual_State,Check_Data_State,S_Chk,Dec_Prod,Out_F,Out_C,Rec,TV_Up,D_FinalOff_Capture,D_FinalTVOff_Capture,Dec_ProdL2,FIX_Prop,D_FinalAppt_Capture,Move_to_Agent bot;
    class C_Int,Q_Cl,C_Cl_Id,C_Age,C_Users,C_Addr,Q_Rec,Q_TV,Q_Appt client;
    class D_Cl_Details_Data,D_Age_Data,D_Users_Data,D_Addr_Data,D_Oval_Data,D_FinalOff_Data,D_FinalTVOff_Data,D_FinalAppt_Data data;