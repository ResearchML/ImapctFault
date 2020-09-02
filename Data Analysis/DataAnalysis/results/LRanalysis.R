
# Assuming the data file is loaded in the R-studio environment
library(dplyr)

# model for all system
df <- nativedata
#head(df)
dfx <- select(df, excessiveInterlangCommunication,Toomuchclustring, ToomuchScattering,UnusedMethodDeclaration, UnusedMethodImplementation, UnusedParameter, AssumingSafeReturnValue, ExcessiveObjects, NotHandlingExceptions, NotCachingObjects, NotSecuringLibraries, HardCodingLibraries, NotUsingRelativePath,MemoryManagementMismatch, LocalReferencesAbuse, LOC,PrevFixing, CodeChurn, InducingFlag)
#head(dfx)
#summary(dfx)
glm.fit <- glm(InducingFlag ~ excessiveInterlangCommunication + Toomuchclustring + ToomuchScattering+UnusedMethodDeclaration+UnusedMethodImplementation+ UnusedParameter+ExcessiveObjects +NotHandlingExceptions+ NotCachingObjects + NotSecuringLibraries+ HardCodingLibraries+ MemoryManagementMismatch + LocalReferencesAbuse+ LOC+PrevFixing, data = dfx, family = binomial(link = "logit"))
summary(glm.fit)


# model per system
df <- nativedata[nativedata$system=="rocksdb",]
#head(df)
dfx <- select(df, excessiveInterlangCommunication,Toomuchclustring, ToomuchScattering,UnusedMethodDeclaration, UnusedMethodImplementation, UnusedParameter, AssumingSafeReturnValue, ExcessiveObjects, NotHandlingExceptions, NotCachingObjects, NotSecuringLibraries, HardCodingLibraries, NotUsingRelativePath,MemoryManagementMismatch, LocalReferencesAbuse, LOC,PrevFixing, CodeChurn, InducingFlag)
#head(dfx)
#summary(dfx)
glm.fit <- glm(InducingFlag ~ excessiveInterlangCommunication + Toomuchclustring + ToomuchScattering+UnusedMethodDeclaration+UnusedMethodImplementation+ UnusedParameter+ ExcessiveObjects +NotHandlingExceptions + NotCachingObjects+NotSecuringLibraries+ HardCodingLibraries+ MemoryManagementMismatch + LocalReferencesAbuse+ LOC+PrevFixing, data = dfx, family = binomial(link = "logit"), control = list(maxit = 100))
summary(glm.fit)


# smell-wise LR model
df <- nativedata[nativedata$system=="javacpp",]
#head(df)
df1 <- df[df$version=="javacpp-1.3",]
#head(df1)
dfx <- select(df1, excessiveInterlangCommunication,Toomuchclustring, ToomuchScattering,UnusedMethodDeclaration, UnusedMethodImplementation, UnusedParameter, AssumingSafeReturnValue, ExcessiveObjects, NotHandlingExceptions, NotCachingObjects, NotSecuringLibraries, HardCodingLibraries, NotUsingRelativePath,MemoryManagementMismatch, LocalReferencesAbuse, LOC,PrevFixing, CodeChurn, InducingFlag)
#head(dfx)
#summary(dfx)
glm.fit <- glm(InducingFlag ~ excessiveInterlangCommunication + Toomuchclustring + ToomuchScattering+UnusedMethodDeclaration+UnusedMethodImplementation+ UnusedParameter+AssumingSafeReturnValue + ExcessiveObjects + NotHandlingExceptions + NotCachingObjects+ NotSecuringLibraries+ HardCodingLibraries+NotUsingRelativePath+ MemoryManagementMismatch + LocalReferencesAbuse+ LOC+PrevFixing+ CodeChurn, data = dfx, family = binomial(link = "logit"))
summary(glm.fit)
