module NewFeatures

using DataFrames

function substrings_in_string(main_string::AbstractString, substrings::Vector{String})
    for substring in substrings
        if occursin(substring, main_string)
            return substring
        end
    end
    return missing
end

function substrings_in_string(main_string::AbstractString, substrings::Vector{Char})
    for substring in substrings
        if occursin(substring, main_string)
            return substring
        end
    end
    return missing
end

substrings_in_string(::Missing, ::Vector{Char}) = missing

function replace_titles(row)
    title = row.Title
    if title in ["Don", "Major", "Capt", "Jonkheer", "Rev", "Col"]
        return "Mr"
    elseif title in ["Countess", "Mme"]
        return "Mrs"
    elseif title in ["Mlle", "Ms"]
        return "Miss"
    elseif title == "Dr"
        return row.Sex == "Male" ? "Mr" : "Mrs"
    else
        return title
    end
end

function create_titles(df::DataFrame)
    title_list = ["Mrs", "Mr", "Master", "Miss", "Major", "Rev",
              "Dr", "Ms", "Mlle", "Col", "Capt", "Mme", "Countess",
              "Don", "Jonkheer"]
    df.Title = [substrings_in_string(name, title_list) for name in df.Name]
    df.Title = replace_titles.(eachrow(df))
    return df
end

function create_family_size(df::DataFrame)
    df.FamilySize = df.SibSp .+ df.Parch
    return df
    
end

function add_new_features(df::DataFrame)
    df = create_titles(df)
    df = create_family_size(df)
    return df
end
    

end
